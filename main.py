"""
Fire Detection Alarm System v4.0 - Production Ready
Developed by Pratik Mishra
A high-performance, real-time computer vision system for fire detection
using OpenCV with advanced multi-stage HSV masking, contour analysis,
flicker motion detection, and smart persistence logic.
Features:
  - Multi-stage HSV detection (aura + core + bright zones)
  - Temporal flicker analysis to reject static orange/red objects
  - Contour shape analysis (aspect ratio, solidity) to reject non-fire shapes
  - Thread-safe audio alarm that plays UNTIL fire is fully gone
  - Auto-generated fallback siren if no audio file is found
  - Optional video recording on detection
  - Interactive HSV calibration mode
  - FPS display and on-screen HUD
Usage:
    python fire_detection.py
    python fire_detection.py --config path/to/config.json
    python fire_detection.py --calibrate
    python fire_detection.py --camera 1
"""
from __future__ import annotations
import argparse
import json
import logging
import math
import os
import struct
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import List, Optional, Tuple
import cv2
import numpy as np
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("pygame not installed — audio alarm disabled. Install with: pip install pygame")

@dataclass

class FireDetectionConfig:
    """Configuration for the fire detection system."""
    min_contour_area: int = 800
    min_core_ratio: float = 0.08
    persistence_threshold: float = 1.5
    flicker_sensitivity: float = 0.08
    flicker_max: float = 0.85
    hsv_aura_lower: np.ndarray = field(
        default_factory = lambda: np.array([0, 120, 150], dtype = np.uint8)
    )
    hsv_aura_upper: np.ndarray = field(
        default_factory = lambda: np.array([25, 255, 255], dtype = np.uint8)
    )
    hsv_core_lower: np.ndarray = field(
        default_factory = lambda: np.array([5, 0, 200], dtype = np.uint8)
    )
    hsv_core_upper: np.ndarray = field(

        default_factory = lambda: np.array([45, 120, 255], dtype = np.uint8)
    )
    hsv_glow_lower: np.ndarray = field(
        default_factory = lambda: np.array([0, 0, 220], dtype = np.uint8)
    )
    hsv_glow_upper: np.ndarray = field(

        default_factory = lambda: np.array([179, 80, 255], dtype = np.uint8)
    )
    alarm_file: str = "alarm.mp3"
    fallback_wav: str = "alarm_generated.wav"
    alarm_volume: float = 0.85
    camera_index: int = 0
    resolution: Tuple[int, int] = (1280, 720)
    fps: int = 30
    window_title: str = "Fire Detection System v4.0  |  Pratik Mishra"
    show_fps: bool = True
    record_on_detection: bool = False
    recording_grace_seconds: int = 10
    output_dir: str = "fire_recordings"

    def validate(self) -> None:
        assert self.min_contour_area >= 100, "min_contour_area must be >= 100"
        assert 0 <= self.min_core_ratio <= 1, "min_core_ratio must be 0–1"
        assert self.persistence_threshold >= 0, "persistence_threshold must be >= 0"
        assert 0 <= self.alarm_volume <= 1, "alarm_volume must be 0–1"

    @classmethod

    def from_json(cls, path: str) -> "FireDetectionConfig":
        with open(path, "r") as f:
            data = json.load(f)
        for key in ("hsv_aura_lower", "hsv_aura_upper",
                    "hsv_core_lower", "hsv_core_upper",
                    "hsv_glow_lower", "hsv_glow_upper"):
            if key in data:
                data[key] = np.array(data[key], dtype = np.uint8)
        obj = cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        obj.validate()
        return obj

    def to_json(self, path: str) -> None:
        data = {
            "min_contour_area": self.min_contour_area,
            "min_core_ratio": self.min_core_ratio,
            "persistence_threshold": self.persistence_threshold,
            "flicker_sensitivity": self.flicker_sensitivity,
            "flicker_max": getattr(self, "flicker_max", 0.60),
            "hsv_aura_lower": self.hsv_aura_lower.tolist(),
            "hsv_aura_upper": self.hsv_aura_upper.tolist(),
            "hsv_core_lower": self.hsv_core_lower.tolist(),
            "hsv_core_upper": self.hsv_core_upper.tolist(),
            "hsv_glow_lower": self.hsv_glow_lower.tolist(),
            "hsv_glow_upper": self.hsv_glow_upper.tolist(),
            "alarm_file": self.alarm_file,
            "fallback_wav": self.fallback_wav,
            "alarm_volume": self.alarm_volume,
            "camera_index": self.camera_index,
            "resolution": list(self.resolution),
            "fps": self.fps,
            "window_title": self.window_title,
            "show_fps": self.show_fps,
            "record_on_detection": self.record_on_detection,
            "recording_grace_seconds": self.recording_grace_seconds,
            "output_dir": self.output_dir,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent = 2)

class AudioManager:
    """Thread-safe singleton audio manager.
    Alarm loops continuously while fire is present and stops only when fire clears.
    """
    _instance: Optional["AudioManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "AudioManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._ready = False
        self._volume = 0.85
        self._queue: Queue[bool] = Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    def initialize(self, config: FireDetectionConfig) -> bool:
        if not PYGAME_AVAILABLE:
            return False
        try:
            pygame.mixer.init(frequency = 44100, size = -16, channels = 2, buffer = 1024)
            self._volume = config.alarm_volume
            pygame.mixer.music.set_volume(self._volume)
            for path in (config.alarm_file, config.fallback_wav):
                if os.path.exists(path):
                    try:
                        pygame.mixer.music.load(path)
                        self._ready = True
                        logging.info(f"Audio loaded: {path}")
                        self._start_thread()
                        return True
                    except pygame.error as e:
                        logging.warning(f"Cannot load {path}: {e}")
            self._generate_siren(config.fallback_wav)
            pygame.mixer.music.load(config.fallback_wav)
            self._ready = True
            logging.info("Using generated siren alarm.")
            self._start_thread()
            return True
        except Exception as e:
            logging.error(f"Audio init failed: {e}")
            return False

    def _generate_siren(self, path: str) -> None:
        """Generate a realistic two-tone emergency siren."""
        sample_rate = 44100
        duration = 2.0
        n = int(sample_rate * duration)
        frames = bytearray()
        for i in range(n):
            t = i / sample_rate
            freq = 880.0 if (t % 0.70) < 0.35 else 660.0
            v = math.sin(2 * math.pi * freq * t)
            v += 0.35 * math.sin(2 * math.pi * freq * 2 * t)
            v += 0.12 * math.sin(2 * math.pi * freq * 3 * t)
            v /= 1.47
            v = max(min(v, 0.97), -0.97)
            left = int(32767 * v * self._volume)
            right = int(32767 * v * 0.88 * self._volume)
            frames += struct.pack("<hh", left, right)
        with wave.open(path, "w") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(bytes(frames))

    def _start_thread(self) -> None:
        self._stop_evt.clear()
        self._thread = threading.Thread(target = self._loop, daemon = True, name = "AudioLoop")
        self._thread.start()

    def _loop(self) -> None:
        playing = False
        while not self._stop_evt.is_set():
            try:
                cmd = self._queue.get(timeout = 0.05)
                if cmd and not playing:
                    pygame.mixer.music.play(-1)
                    playing = True
                elif not cmd and playing:
                    pygame.mixer.music.stop()
                    playing = False
            except Empty:
                pass
            except Exception as e:
                logging.error(f"Audio loop error: {e}")

    def alarm_on(self) -> None:
        """Signal: fire is present — keep alarm running."""
        if self._ready:
            self._queue.put(True)

    def alarm_off(self) -> None:
        """Signal: fire is gone — stop alarm."""
        if self._ready:
            self._queue.put(False)

    def shutdown(self) -> None:
        self._stop_evt.set()
        if self._ready:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except Exception:
                pass

class VideoRecorder:
    """Optional video recorder that saves footage when fire is detected."""

    def __init__(self, config: FireDetectionConfig) -> None:
        self._cfg = config
        self._writer: Optional[cv2.VideoWriter] = None
        self._recording = False
        self._last_fire_time: float = 0.0
        self._filepath: Optional[str] = None
        if config.record_on_detection:
            Path(config.output_dir).mkdir(parents = True, exist_ok = True)

    def notify(self, fire_active: bool, frame: np.ndarray) -> None:
        if not self._cfg.record_on_detection:
            return
        if fire_active:
            self._last_fire_time = time.time()
            if not self._recording:
                self._start(frame.shape)
        if self._recording:
            self._writer.write(frame)
            grace = time.time() - self._last_fire_time
            if not fire_active and grace > self._cfg.recording_grace_seconds:
                self._stop()

    def _start(self, shape: Tuple) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._filepath = os.path.join(self._cfg.output_dir, f"fire_{ts}.mp4")
        h, w = shape[:2]
        self._writer = cv2.VideoWriter(
            self._filepath,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._cfg.fps,
            (w, h),
        )
        self._recording = True
        logging.info(f"Recording started: {self._filepath}")

    def _stop(self) -> None:
        if self._writer:
            self._writer.release()
            self._writer = None
        self._recording = False
        logging.info(f"Recording saved: {self._filepath}")

    def close(self) -> None:
        self._stop()

class FireDetector:
    """
    Multi-stage fire detector:
      1. HSV masking (aura + core + glow layers)
      2. Morphological cleanup
      3. Contour filtering (area, core ratio, aspect ratio, solidity)
      4. Temporal flicker analysis (real fire flickers; static objects don't)
      5. Persistence gate (alarm only after N seconds of confirmed detection)
    """
    FLICKER_HISTORY = 8

    def __init__(self, config: FireDetectionConfig) -> None:
        self._cfg = config
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._fire_first_seen: Optional[float] = None
        self._fire_last_seen: float = 0.0
        self._alarm_active = False
        self._prev_mask: Optional[np.ndarray] = None
        self._flicker_scores: List[float] = []
        self._fps_times: List[float] = []
        self._last_t = time.time()

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Process one BGR frame.  Returns (annotated_frame, alarm_active)."""
        frame = cv2.flip(frame, 1)
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask_aura = cv2.inRange(hsv, self._cfg.hsv_aura_lower, self._cfg.hsv_aura_upper)
        mask_core = cv2.inRange(hsv, self._cfg.hsv_core_lower, self._cfg.hsv_core_upper)
        mask_glow = cv2.inRange(hsv, self._cfg.hsv_glow_lower, self._cfg.hsv_glow_upper)
        mask_hot = cv2.bitwise_or(mask_core, mask_glow)
        mask_aura = cv2.morphologyEx(mask_aura, cv2.MORPH_CLOSE, self._kernel_close, iterations = 2)
        mask_aura = cv2.morphologyEx(mask_aura, cv2.MORPH_OPEN,  self._kernel_open,  iterations = 1)
        if self._prev_mask is None or self._prev_mask.shape != mask_aura.shape:
            self._prev_mask = mask_aura.copy()
            diff_mask = np.ones_like(mask_aura) * 255
        else:
            diff_mask = cv2.absdiff(mask_aura, self._prev_mask)
            self._prev_mask = mask_aura.copy()
        self._flicker_scores.append(1.0)
        if len(self._flicker_scores) > self.FLICKER_HISTORY:
            self._flicker_scores.pop(0)
        contours, _ = cv2.findContours(mask_aura, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fire_found = False
        boxes: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._cfg.min_contour_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / (bh + 1e-6)
            if aspect > 2.5:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = area / (hull_area + 1e-6)
            if solidity > 0.95:
                continue
            roi_hot = mask_hot[y:y + bh, x:x + bw]
            hot_pixels = cv2.countNonZero(roi_hot)
            if hot_pixels < bw * bh * self._cfg.min_core_ratio:
                continue
            h_f, w_f = diff_mask.shape
            pad = 5
            y_min, y_max = max(0, y - pad), min(h_f, y + bh + pad)
            x_min, x_max = max(0, x - pad), min(w_f, x + bw + pad)
            roi_diff = diff_mask[y_min:y_max, x_min:x_max]
            roi_mask = mask_aura[y_min:y_max, x_min:x_max]
            local_total = cv2.countNonZero(roi_mask) + 1e-6
            local_flicker = float(cv2.countNonZero(roi_diff)) / local_total
            if len(self._flicker_scores) >= self.FLICKER_HISTORY:
                if local_flicker < self._cfg.flicker_sensitivity:
                    continue
                fmax = getattr(self._cfg, 'flicker_max', 0.60)
                if local_flicker > fmax:
                    continue
            fire_found = True
            boxes.append((x, y, bw, bh))
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 70, 255), 2)
            label_y = max(y - 10, 20)
            cv2.putText(
                frame, "FIRE", (x, label_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 70, 255), 2, cv2.LINE_AA,
            )
            overlay_region = frame[y:y + bh, x:x + bw]
            tint = overlay_region.copy()
            tint[:] = (0, 0, 200)
            frame[y:y + bh, x:x + bw] = cv2.addWeighted(overlay_region, 0.75, tint, 0.25, 0)
        alarm = self._update_persistence(fire_found)
        frame = self._draw_hud(frame, alarm)
        return frame, alarm

    def _update_persistence(self, fire_detected: bool) -> bool:
        now = time.time()
        if fire_detected:
            self._fire_last_seen = now
            if self._fire_first_seen is None:
                self._fire_first_seen = now
            if (now - self._fire_first_seen) >= self._cfg.persistence_threshold:
                self._alarm_active = True
        else:
            if (now - self._fire_last_seen) > 2.0:
                self._fire_first_seen = None
                self._alarm_active = False
        return self._alarm_active

    def _draw_hud(self, frame: np.ndarray, alarm: bool) -> np.ndarray:
        h, w = frame.shape[:2]
        bar_h = 54
        bar = frame.copy()
        cv2.rectangle(bar, (0, h - bar_h), (w, h), (10, 10, 10), -1)
        frame = cv2.addWeighted(bar, 0.45, frame, 0.55, 0)
        if alarm:
            status_text = "*** FIRE DETECTED! EVACUATE! ***"
            status_color = (0, 50, 255)
            if int(time.time() * 5) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 230), 12)
        else:
            status_text = "  MONITORING  "
            status_color = (50, 210, 50)
        cv2.putText(
            frame, status_text, (16, h - 16),
            cv2.FONT_HERSHEY_DUPLEX, 0.75, status_color, 2, cv2.LINE_AA,
        )
        credit = "Pratik Mishra  |  v4.0"
        (tw, _), _ = cv2.getTextSize(credit, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.putText(
            frame, credit, (w - tw - 16, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 160, 160), 1, cv2.LINE_AA,
        )
        if self._cfg.show_fps:
            now = time.time()
            dt = now - self._last_t
            self._last_t = now
            if dt > 0:
                self._fps_times.append(1.0 / dt)
            if len(self._fps_times) > 30:
                self._fps_times.pop(0)
            avg_fps = sum(self._fps_times) / len(self._fps_times) if self._fps_times else 0
            cv2.putText(
                frame, f"FPS: {avg_fps:.1f}", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA,
            )
        return frame

def run_calibration(camera_index: int = 0) -> None:
    """Interactive HSV calibration helper."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logging.error("Cannot open camera for calibration.")
        return
    cv2.namedWindow("HSV Calibration", cv2.WINDOW_NORMAL)
    for name, default, maxv in [
        ("H Lo", 0,   179), ("H Hi", 25,  179),
        ("S Lo", 120, 255), ("S Hi", 255, 255),
        ("V Lo", 120, 255), ("V Hi", 255, 255),
    ]:
        cv2.createTrackbar(name, "HSV Calibration", default, maxv, lambda _: None)
    print("Calibration mode — point camera at fire.")
    print("Adjust sliders until only fire is white in the mask panel.")
    print("  s → save config   q → quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2HSV)
        lo = np.array([
            cv2.getTrackbarPos("H Lo", "HSV Calibration"),
            cv2.getTrackbarPos("S Lo", "HSV Calibration"),
            cv2.getTrackbarPos("V Lo", "HSV Calibration"),
        ], dtype = np.uint8)
        hi = np.array([
            cv2.getTrackbarPos("H Hi", "HSV Calibration"),
            cv2.getTrackbarPos("S Hi", "HSV Calibration"),
            cv2.getTrackbarPos("V Hi", "HSV Calibration"),
        ], dtype = np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        result = cv2.bitwise_and(frame, frame, mask = mask)
        panel = np.hstack([
            frame,
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            result,
        ])
        cv2.imshow("HSV Calibration", panel)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            cfg = FireDetectionConfig(
                hsv_aura_lower = lo,
                hsv_aura_upper = hi,
            )
            cfg.to_json("calibrated_config.json")
            print("Saved → calibrated_config.json")
    cap.release()
    cv2.destroyAllWindows()

class FireDetectionSystem:
    """Wires detector, audio, and recorder together."""

    def __init__(self, config: FireDetectionConfig) -> None:
        self._cfg = config
        self._detector = FireDetector(config)
        self._audio = AudioManager()
        self._recorder = VideoRecorder(config)
        self._running = False
        self._prev_alarm = False

    def run(self) -> None:
        cap = cv2.VideoCapture(self._cfg.camera_index)
        if not cap.isOpened():
            logging.error(f"Cannot open camera index {self._cfg.camera_index}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cfg.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.resolution[1])
        cap.set(cv2.CAP_PROP_FPS,          self._cfg.fps)
        audio_ok = self._audio.initialize(self._cfg)
        if not audio_ok:
            logging.warning("No audio — visual alarm only.")
        self._running = True
        logging.info(f"{self._cfg.window_title} started. Press 'q' to quit.")
        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Frame read failed — retrying…")
                    time.sleep(0.05)
                    continue
                annotated, alarm = self._detector.process(frame)
                if alarm and not self._prev_alarm:
                    self._audio.alarm_on()
                    logging.warning("FIRE DETECTED — alarm on")
                elif not alarm and self._prev_alarm:
                    self._audio.alarm_off()
                    logging.info("Fire cleared — alarm off")
                self._prev_alarm = alarm
                if alarm:
                    self._audio.alarm_on()
                self._recorder.notify(alarm, annotated)
                cv2.imshow(self._cfg.window_title, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            logging.info("Interrupted by user.")
        finally:
            self._shutdown(cap)

    def _shutdown(self, cap: cv2.VideoCapture) -> None:
        self._running = False
        self._audio.alarm_off()
        time.sleep(0.15)
        self._audio.shutdown()
        self._recorder.close()
        cap.release()
        cv2.destroyAllWindows()
        logging.info("System shut down cleanly.")

def setup_logging() -> None:
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt = "%H:%M:%S",
    )

def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        description = "Fire Detection System v4.0  |  Pratik Mishra",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",    "-c",   help = "Path to JSON config file")
    parser.add_argument("--calibrate", action = "store_true", help = "Run HSV calibration mode")
    parser.add_argument("--camera",    "-cam", type = int, default = 0, help = "Camera index")
    args = parser.parse_args()
    if args.calibrate:
        run_calibration(args.camera)
        return 0
    if args.config and os.path.exists(args.config):
        config = FireDetectionConfig.from_json(args.config)
        logging.info(f"Config loaded from {args.config}")
    else:
        if args.config:
            logging.warning(f"Config not found: {args.config} — using defaults")
        config = FireDetectionConfig(camera_index = args.camera)
    config.validate()
    system = FireDetectionSystem(config)
    system.run()
    return 0
if __name__ == "__main__":
    sys.exit(main())
