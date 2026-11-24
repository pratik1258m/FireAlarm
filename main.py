"""
Fire Detection Alarm System v5.4 - Production Ready
Developed by Pratik Mishra

Uses a research-backed multi-color-space + temporal pipeline:
1. Chen et al. (2004) RGB rule     - R >= G > B (rejects red cloth/pillows where G ~= B)
2. Celik & Demirel (2009) YCbCr    - adaptive to room lighting (rejects bright beige walls)
3. White-hot core + halo check     - catches small, overexposed match/lighter flames
4. MOG2 motion + edge frame-diff   - rejects static objects, freezes learning during alarm
5. Temporal flicker tracking       - per-blob area, shape (IoU), and luminance pulsing
6. Hysteresis confidence gate      - prevents false triggers and stuck alarms

Usage:
    python main.py
    python main.py --config path/to/config.json
    python main.py --camera 1
    python main.py --debug
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
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("pygame not installed — audio alarm disabled. Install with: pip install pygame")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(min(high, max(low, value)))


@dataclass
class FireDetectionConfig:
    """Configuration for the fire detection system."""

    # --- Chen et al. (RGB) + Celik & Demirel (YCbCr) thresholds ---
    r_min: int = 215                 # fire is luminous in red (rejects beige wall & skin)
    gb_diff_min: int = 25            # G - B >= 25 rejects pure red cloth/pillows
    ycbcr_tau: int = 45              # |Cr - Cb| >= 45 rejects warm wall tint
    hot_y_min: int = 242             # white-hot flame core luminance

    # --- blob filtering (evaluated at process_width) ---
    min_contour_area: int = 20       # allows small match/lighter flames at 640px
    min_area_ratio: float = 0.00002
    min_hot_area: int = 15
    max_hot_area_ratio: float = 0.04  # rejects large bright walls/windows
    min_glow_ratio: float = 0.08     # fire-colored halo required around a white-hot core
    min_flicker: float = 0.18        # fire flickers; static objects do not
    max_area_ratio: float = 0.65
    max_aspect: float = 3.5
    min_solidity: float = 0.30
    max_solidity: float = 0.99

    # --- confidence decision ---
    min_fire_score: float = 0.62          # calibrated for small flames (match/lighter)
    clear_score: float = 0.52             # show suspects above 52%
    score_smoothing: float = 0.50         # responsive without jitter
    warmup_frames: int = 4                # 4 frames of temporal data before scoring
    persistence_threshold: float = 0.6    # sustained detection without multi-second lag
    persistence_grace: float = 0.40       # won't reset if 1 frame dips
    clear_delay: float = 3.0              # hold clear confirmation before stopping alarm

    # --- motion filtering ---
    use_bg_subtraction: bool = True       # restrict candidates to moving regions only
    bg_history: int = 300                 # MOG2 background model history (frames)
    bg_var_threshold: float = 32.0        # MOG2 pixel variance threshold
    motion_min_overlap: float = 0.18      # fraction of blob pixels that must be in motion
    use_optical_flow: bool = True         # score per-blob edge frame-diff flicker (fast)

    # --- audio ---
    alarm_file: str = "alarm.mp3"
    fallback_wav: str = "alarm_generated.wav"
    alarm_volume: float = 0.85

    # --- runtime ---
    camera_index: int = 0
    resolution: Tuple[int, int] = (1280, 720)
    process_width: int = 640              # downscale width for 25-30 FPS
    fps: int = 30
    window_title: str = "Fire Detection System v5.4  |  Pratik Mishra"
    show_fps: bool = True
    show_masks: bool = False
    record_on_detection: bool = False
    recording_grace_seconds: int = 10
    output_dir: str = "fire_recordings"

    def validate(self) -> None:
        assert self.min_contour_area >= 10, "min_contour_area must be >= 10"
        assert 0 <= self.min_area_ratio <= 1, "min_area_ratio must be 0-1"
        assert 0 < self.max_area_ratio <= 1, "max_area_ratio must be 0-1"
        assert 0 <= self.persistence_threshold, "persistence_threshold must be >= 0"
        assert 0 <= self.alarm_volume <= 1, "alarm_volume must be 0-1"
        assert 0 < self.min_fire_score <= 1, "min_fire_score must be 0-1"
        assert 0 <= self.clear_score <= self.min_fire_score, "clear_score must be <= min_fire_score"

    @classmethod
    def from_json(cls, path: str) -> "FireDetectionConfig":
        with open(path, "r") as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        if "resolution" in filtered and isinstance(filtered["resolution"], list):
            filtered["resolution"] = tuple(filtered["resolution"])
        obj = cls(**filtered)
        obj.validate()
        return obj

    def to_json(self, path: str) -> None:
        data = {}
        for f in fields(self):
            value = getattr(self, f.name)
            data[f.name] = list(value) if isinstance(value, tuple) else value
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class AudioManager:
    """Thread-safe singleton audio manager."""

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
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
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
        self._thread = threading.Thread(target=self._loop, daemon=True, name="AudioLoop")
        self._thread.start()

    def _loop(self) -> None:
        playing = False
        while not self._stop_evt.is_set():
            try:
                cmd = self._queue.get(timeout=0.05)
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
        if self._ready:
            self._queue.put(True)

    def alarm_off(self) -> None:
        if self._ready:
            self._queue.put(False)

    def shutdown(self) -> None:
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
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
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)

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
            self._filepath, cv2.VideoWriter_fourcc(*"mp4v"), self._cfg.fps, (w, h)
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


@dataclass
class _Track:
    """Per-blob temporal history used to measure real fire flicker."""

    track_id: int
    cx: float
    cy: float
    area: int
    bbox: Tuple[int, int, int, int]
    value: float = 0.0
    score: float = 0.0
    misses: int = 0
    frames: int = 1

    def __post_init__(self) -> None:
        self.area_hist: deque = deque(maxlen=FireDetector.HISTORY)
        self.iou_hist: deque = deque(maxlen=FireDetector.HISTORY)
        self.value_hist: deque = deque(maxlen=FireDetector.HISTORY)
        self.cy_hist: deque = deque(maxlen=FireDetector.HISTORY)
        self.area_hist.append(float(self.area))
        self.value_hist.append(float(self.value))
        self.cy_hist.append(float(self.cy))


class FireDetector:
    """
    Research-backed fire detector combining:
    - Chen et al. (2004) RGB channel ordering (R >= G > B)
    - Celik & Demirel (2009) adaptive YCbCr fire chrominance model
    - MOG2 background subtraction + per-blob edge frame-diff flicker
    - Per-blob temporal flicker (area + IoU + luminance oscillation)
    """

    HISTORY = 16
    WEIGHTS = {"core": 0.26, "chroma": 0.22, "temporal": 0.26, "variation": 0.08, "flow": 0.18}

    def __init__(self, config: FireDetectionConfig) -> None:
        self._cfg = config
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._prev_mask: Optional[np.ndarray] = None
        self._tracks: Dict[int, _Track] = {}
        self._next_track_id = 0
        self._fire_first_seen: Optional[float] = None
        self._fire_last_seen: float = 0.0
        self._alarm_active = False
        self._last_score = 0.0
        self._last_flicker = 0.0
        self._last_t = time.time()
        self._fps_times: List[float] = []
        self.debug_view = config.show_masks
        self.debug_image: Optional[np.ndarray] = None
        self.last_candidates: List[dict] = []

        self._bg_sub: Optional[cv2.BackgroundSubtractor] = None
        if config.use_bg_subtraction:
            self._bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=config.bg_history,
                varThreshold=config.bg_var_threshold,
                detectShadows=False,
            )
        self._prev_gray: Optional[np.ndarray] = None

    # --------------------------------------------------------- segmentation
    def _segment(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (mask_fire, mask_hot, mask_halo, ycrcb)."""
        cfg = self._cfg
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y_ch, cr_ch, cb_ch = cv2.split(ycrcb)
        b_ch, g_ch, r_ch = cv2.split(bgr)

        # Scene-adaptive means (Celik & Demirel, 2009)
        y_mean = float(y_ch.mean())
        cr_mean = float(cr_ch.mean())
        cb_mean = float(cb_ch.mean())

        y_f = y_ch.astype(np.int16)
        cr_f = cr_ch.astype(np.int16)
        cb_f = cb_ch.astype(np.int16)
        r_f = r_ch.astype(np.int16)
        g_f = g_ch.astype(np.int16)
        b_f = b_ch.astype(np.int16)

        # 1. Chen et al. RGB rule: R >= G > B with G - B gap (rejects red cloth)
        rgb_fire = (r_f >= cfg.r_min) & (r_f >= g_f) & ((g_f - b_f) >= cfg.gb_diff_min)

        # 2. Celik & Demirel YCbCr rule: brighter than room + warm chrominance
        ycrcb_fire = (
            (y_f > cb_f)
            & (cr_f > cb_f)
            & (y_f > y_mean)
            & (cr_f > cr_mean)
            & (cb_f < cb_mean)
            & ((cr_f - cb_f) >= cfg.ycbcr_tau)
        )

        mask_fire = (rgb_fire & ycrcb_fire).astype(np.uint8) * 255
        mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_OPEN, self._kernel_open, iterations=1)
        mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_CLOSE, self._kernel_close, iterations=1)

        # Relaxed fire glow (halo) used to validate small overexposed match cores
        halo = (
            (r_f >= max(160, cfg.r_min - 30))
            & (r_f >= g_f)
            & ((g_f - b_f) >= max(12, cfg.gb_diff_min - 8))
            & ((cr_f - cb_f) >= max(22, cfg.ycbcr_tau - 13))
            & (y_f > y_mean)
        )
        mask_halo = halo.astype(np.uint8) * 255

        # White-hot core: very high luminance + warm tint (R >= B)
        hot = (y_f >= cfg.hot_y_min) & (r_f >= g_f - 10) & (r_f >= b_f)
        mask_hot = hot.astype(np.uint8) * 255

        return mask_fire, mask_hot, mask_halo, ycrcb

    @staticmethod
    def _compact_hot_mask(mask_hot: np.ndarray, max_px: int) -> np.ndarray:
        """Keep only compact pale blobs (rejects bright walls/windows)."""
        out = np.zeros_like(mask_hot)
        contours, _ = cv2.findContours(mask_hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if cv2.countNonZero(mask_hot[y:y + bh, x:x + bw]) <= max_px:
                cv2.drawContours(out, [cnt], -1, 255, -1)
        return out

    # ------------------------------------------------------------- candidates
    def _build_candidate(
        self,
        cnt: np.ndarray,
        ycrcb: np.ndarray,
        src_mask: np.ndarray,
        mask_halo: np.ndarray,
        combined: np.ndarray,
        lo_area: int,
        hi_area: int,
        is_hot: bool,
    ) -> Optional[dict]:
        cfg = self._cfg
        h, w = src_mask.shape[:2]
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 3 or bh < 3:
            return None
        roi_mask = src_mask[y:y + bh, x:x + bw]
        area = int(cv2.countNonZero(roi_mask))
        if area <= 0 or area < lo_area or area > hi_area:
            return None
        if bw / (bh + 1e-6) > cfg.max_aspect:
            return None

        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = cv2.contourArea(cnt) / (hull_area + 1e-6)
        if solidity < cfg.min_solidity or solidity > cfg.max_solidity:
            return None

        # Measure fire-colored halo around the blob (padded so dilation isn't clipped)
        reach = max(7, int(0.55 * max(bw, bh)))
        reach += 1 if reach % 2 == 0 else 0
        r_pad = reach // 2
        ry0, ry1 = max(0, y - r_pad), min(h, y + bh + r_pad)
        rx0, rx1 = max(0, x - r_pad), min(w, x + bw + r_pad)
        padded_mask = src_mask[ry0:ry1, rx0:rx1]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))
        ring = cv2.subtract(cv2.dilate(padded_mask, kernel), padded_mask)
        ring_px = cv2.countNonZero(ring)
        glow = 0.0
        if ring_px > 10:
            core_ring = cv2.bitwise_and(ring, mask_halo[ry0:ry1, rx0:rx1])
            glow = cv2.countNonZero(core_ring) / float(ring_px)

        if is_hot and glow < cfg.min_glow_ratio:
            return None

        if is_hot:
            pad = max(4, int(0.35 * max(bw, bh)))
            y0, y1 = max(0, y - pad), min(h, y + bh + pad)
            x0, x1 = max(0, x - pad), min(w, x + bw + pad)
            ycrcb_roi = ycrcb[y0:y1, x0:x1]
            halo_roi = mask_halo[y0:y1, x0:x1]
            blob_roi = combined[y0:y1, x0:x1]
        else:
            ycrcb_roi = ycrcb[y:y + bh, x:x + bw]
            halo_roi = mask_halo[y:y + bh, x:x + bw]
            blob_roi = roi_mask

        y_vals = ycrcb_roi[:, :, 0]
        glow_px = y_vals[halo_roi > 0]
        if glow_px.size < 10:
            glow_px = y_vals[blob_roi > 0]
        mean_y = float(glow_px.mean()) if glow_px.size else 0.0

        moments = cv2.moments(cnt)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx, cy = x + bw / 2.0, y + bh / 2.0

        return {
            "bbox": (x, y, bw, bh),
            "source": "hot" if is_hot else "fire",
            "area": area,
            "centroid": (cx, cy),
            "roi_mask": roi_mask,
            "mean_v": mean_y,
            "s_core": self._score_core(ycrcb_roi, blob_roi, glow),
            "s_chroma": self._score_chroma(ycrcb_roi, blob_roi, halo_roi),
            "s_variation": self._score_variation(ycrcb_roi, blob_roi, halo_roi),
            "s_flow": 0.35,
            "score": 0.0,
        }

    @staticmethod
    def _contained(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> bool:
        ix, iy, iw, ih = inner
        ox, oy, ow, oh = outer
        x0, y0 = max(ix, ox), max(iy, oy)
        x1, y1 = min(ix + iw, ox + ow), min(iy + ih, oy + oh)
        if x1 <= x0 or y1 <= y0:
            return False
        return ((x1 - x0) * (y1 - y0)) / float(max(1, iw * ih)) > 0.85

    # ------------------------------------------------------------- sub-scores
    def _score_core(self, ycrcb_roi: np.ndarray, blob_roi: np.ndarray, glow: float) -> float:
        """Luminance peak check: real flames are bright light sources."""
        y_vals = ycrcb_roi[:, :, 0][blob_roi > 0]
        if y_vals.size < 8:
            return 0.0
        p95 = float(np.percentile(y_vals, 95))
        if p95 < 200.0:
            return 0.0
        score = _clamp((p95 - 195.0) / 50.0)
        hot_frac = float((y_vals >= self._cfg.hot_y_min).mean())
        if hot_frac > 0.90 and glow < 0.12:
            score *= 0.30  # uniformly blown-out white object with no fire halo
        return _clamp(score)

    def _score_chroma(self, ycrcb_roi: np.ndarray, blob_roi: np.ndarray, halo_roi: np.ndarray) -> float:
        """Celik Cr - Cb fire chrominance strength."""
        sel = (blob_roi > 0) | (halo_roi > 0)
        if int(sel.sum()) < 8:
            return 0.0
        cr = ycrcb_roi[:, :, 1][sel].astype(np.float32)
        cb = ycrcb_roi[:, :, 2][sel].astype(np.float32)
        diff = cr - cb
        p80 = float(np.percentile(diff, 80))
        return _clamp((p80 - 20.0) / 55.0)

    def _score_variation(self, ycrcb_roi: np.ndarray, blob_roi: np.ndarray, halo_roi: np.ndarray) -> float:
        """Flames have internal luminance and chrominance gradients; flat objects don't."""
        sel = (blob_roi > 0) | (halo_roi > 0)
        if int(sel.sum()) < 8:
            return 0.0
        y_v = ycrcb_roi[:, :, 0][sel].astype(np.float32)
        cr_v = ycrcb_roi[:, :, 1][sel].astype(np.float32)
        return _clamp(0.5 * (float(y_v.std()) / 30.0) + 0.5 * (float(cr_v.std()) / 15.0))

    def _score_temporal(self, track: _Track) -> float:
        """Fire flickers in shape/area and/or luminance.

        Uses a weighted sum instead of a product so overexposed white-hot
        flames (whose core clips at 255) still score strongly on shape flicker.
        """
        if track.frames < self._cfg.warmup_frames:
            return 0.0
        areas = np.array(track.area_hist, dtype=np.float32)
        mean_area = float(areas.mean())
        if mean_area <= 1.0:
            return 0.0
        area_score = _clamp(float(areas.std() / mean_area) / 0.08)
        if track.iou_hist:
            shape_score = _clamp((1.0 - float(np.mean(track.iou_hist))) / 0.22)
        else:
            shape_score = 0.0
        shape_change = 0.55 * area_score + 0.45 * shape_score

        values = np.array(track.value_hist, dtype=np.float32)
        mean_value = float(values.mean())
        brightness_change = 0.0
        if mean_value > 1.0:
            brightness_change = _clamp(float(values.std() / mean_value) / 0.025)

        return _clamp(0.70 * shape_change + 0.30 * brightness_change)

    @staticmethod
    def _score_frame_diff(
        gray_curr: Optional[np.ndarray],
        gray_prev: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
    ) -> float:
        """Per-blob edge frame-differencing flicker score.

        Pads the bbox by 2px so it captures the flickering flame boundary
        without pulling in unrelated moving objects nearby.
        """
        if gray_curr is None or gray_prev is None:
            return 0.35
        x, y, bw, bh = bbox
        h, w = gray_curr.shape
        pad = 2
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        if y1 <= y0 or x1 <= x0:
            return 0.35
        curr_roi = gray_curr[y0:y1, x0:x1].astype(np.float32)
        prev_roi = gray_prev[y0:y1, x0:x1].astype(np.float32)
        diff = np.abs(curr_roi - prev_roi)
        edge_diff = float(np.percentile(diff, 85))
        return _clamp(edge_diff / 22.0)

    # ------------------------------------------------------------- tracking
    def _match_tracks(self, candidates: List[dict], now: float) -> List[dict]:
        for track in self._tracks.values():
            track.misses += 1

        matched_ids: Set[int] = set()
        for cand in candidates:
            cx, cy = cand["centroid"]
            best: Optional[_Track] = None
            best_dist = float("inf")
            for track in self._tracks.values():
                if track.track_id in matched_ids:
                    continue
                smaller = max(1, min(track.area, cand["area"]))
                larger = max(track.area, cand["area"])
                if larger / float(smaller) > 8.0:
                    continue
                dist = math.hypot(track.cx - cx, track.cy - cy)
                reach = max(35.0, 0.75 * math.sqrt(max(track.area, 1)))
                if dist < reach and dist < best_dist:
                    best, best_dist = track, dist

            x, y, bw, bh = cand["bbox"]
            if best is None:
                track = _Track(
                    self._next_track_id, cx, cy, cand["area"], cand["bbox"],
                    value=cand.get("mean_v", 0.0),
                )
                self._next_track_id += 1
                self._tracks[track.track_id] = track
                matched_ids.add(track.track_id)
                cand["track"] = track
            else:
                matched_ids.add(best.track_id)
                prev_crop = None
                if self._prev_mask is not None:
                    ph, pw = self._prev_mask.shape[:2]
                    y0, y1 = max(0, y), min(ph, y + bh)
                    x0, x1 = max(0, x), min(pw, x + bw)
                    if y1 > y0 and x1 > x0:
                        prev_crop = self._prev_mask[y0:y1, x0:x1]

                best.misses = 0
                best.frames += 1
                best.cx, best.cy = cx, cy
                best.area = cand["area"]
                best.bbox = cand["bbox"]
                best.area_hist.append(float(cand["area"]))
                best.value_hist.append(float(cand.get("mean_v", 0.0)))
                best.cy_hist.append(float(cy))
                if prev_crop is not None and prev_crop.shape == cand["roi_mask"].shape:
                    inter = cv2.countNonZero(cv2.bitwise_and(cand["roi_mask"], prev_crop))
                    union = cv2.countNonZero(cv2.bitwise_or(cand["roi_mask"], prev_crop))
                    if union > 0:
                        best.iou_hist.append(inter / float(union))
                cand["track"] = best

        dead = [tid for tid, t in self._tracks.items() if t.misses > 12]
        for tid in dead:
            del self._tracks[tid]
        return candidates

    def _update_confidence(self, candidates: List[dict]) -> Tuple[float, float]:
        for track in self._tracks.values():
            if track.misses > 0:
                track.score *= 0.72

        best = 0.0
        best_flicker = 0.0
        for cand in candidates:
            track: _Track = cand["track"]
            flicker = self._score_temporal(track)
            flow = cand.get("s_flow", 0.35)
            effective_flicker = max(flicker, 0.75 * flow)
            if cand["s_core"] < 0.25 or cand["s_chroma"] < 0.25:
                instant = 0.0
            else:
                instant = (
                    self.WEIGHTS["core"] * cand["s_core"]
                    + self.WEIGHTS["chroma"] * cand["s_chroma"]
                    + self.WEIGHTS["temporal"] * flicker
                    + self.WEIGHTS["variation"] * cand["s_variation"]
                    + self.WEIGHTS["flow"] * flow
                )
            if track.frames <= 1:
                track.score = instant
            else:
                alpha = self._cfg.score_smoothing
                track.score = (1.0 - alpha) * track.score + alpha * instant
            cand["score"] = track.score
            if track.score > best:
                best = track.score
                best_flicker = effective_flicker
        return best, best_flicker

    # ------------------------------------------------------------- main entry
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        now = time.time()
        h_orig, w_orig = frame.shape[:2]
        target_w = self._cfg.process_width
        scale = target_w / float(w_orig) if w_orig > target_w else 1.0
        proc = cv2.resize(frame, (int(w_orig * scale), int(h_orig * scale))) if scale < 1.0 else frame
        blurred = cv2.GaussianBlur(proc, (3, 3), 0)

        # --- Background subtraction: freeze learning while alarm is active ---
        if self._bg_sub is not None:
            lr = 0.0 if self._alarm_active else 0.002
            fg_mask = self._bg_sub.apply(blurred, learningRate=lr)
            fg_dilated = cv2.dilate(fg_mask, self._kernel_close, iterations=2)
        else:
            fg_dilated = None

        gray_curr = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        gray_prev = self._prev_gray
        self._prev_gray = gray_curr

        mask_fire, raw_hot, mask_halo, ycrcb = self._segment(blurred)

        h, w = mask_fire.shape[:2]
        frame_area = float(h * w)
        min_px = max(self._cfg.min_contour_area, int(self._cfg.min_area_ratio * frame_area))
        max_px = int(self._cfg.max_area_ratio * frame_area)
        max_hot_px = int(self._cfg.max_hot_area_ratio * frame_area)

        mask_hot = self._compact_hot_mask(raw_hot, max_hot_px)
        combined = cv2.bitwise_or(mask_fire, mask_hot)

        candidates: List[dict] = []
        for src_mask, lo_area, hi_area, is_hot in (
            (mask_fire, min_px, max_px, False),
            (mask_hot, self._cfg.min_hot_area, max_hot_px, True),
        ):
            contours, _ = cv2.findContours(src_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cand = self._build_candidate(
                    cnt, ycrcb, src_mask, mask_halo, combined, lo_area, hi_area, is_hot
                )
                if cand is not None:
                    candidates.append(cand)

        # --- Motion gate: drop blobs that aren't in a moving region (MOG2) ---
        if fg_dilated is not None and self._cfg.motion_min_overlap > 0:
            gated = []
            for cand in candidates:
                x, y, bw, bh = cand["bbox"]
                fg_roi = fg_dilated[y : y + bh, x : x + bw]
                total = bw * bh
                if total > 0 and cv2.countNonZero(fg_roi) / float(total) >= self._cfg.motion_min_overlap:
                    gated.append(cand)
            candidates = gated

        self.last_candidates = candidates

        # --- Per-candidate edge frame-diff flicker score ---
        if self._cfg.use_optical_flow:
            for cand in candidates:
                cand["s_flow"] = self._score_frame_diff(gray_curr, gray_prev, cand["bbox"])

        candidates = self._match_tracks(candidates, now)
        confidence, flicker = self._update_confidence(candidates)
        self._last_score = confidence
        self._last_flicker = flicker
        self._prev_mask = combined.copy()

        fire_confirmed = confidence >= self._cfg.min_fire_score and flicker >= self._cfg.min_flicker
        alarm = self._update_persistence(fire_confirmed, now)

        annotated = frame.copy()
        drawn: List[Tuple[int, int, int, int]] = []
        inv = 1.0 / scale
        for cand in sorted(candidates, key=lambda c: -c.get("score", 0.0)):
            score = cand.get("score", 0.0)
            if score < self._cfg.clear_score or cand["track"].frames < self._cfg.warmup_frames:
                continue
            x, y, bw, bh = cand["bbox"]
            if scale < 1.0:
                x, y, bw, bh = int(x * inv), int(y * inv), int(bw * inv), int(bh * inv)
            if any(self._contained((x, y, bw, bh), box) for box in drawn):
                continue
            drawn.append((x, y, bw, bh))
            fired = score >= self._cfg.min_fire_score
            colour = (0, 70, 255) if fired else (0, 190, 255)
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), colour, 2)
            label = f"{'FIRE' if fired else 'SUSPECT'} {score * 100:.0f}%"
            cv2.putText(
                annotated, label, (x, max(y - 8, 18)),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, colour, 2, cv2.LINE_AA,
            )
            if fired:
                roi = annotated[y:y + bh, x:x + bw]
                tint = roi.copy()
                tint[:] = (0, 0, 200)
                annotated[y:y + bh, x:x + bw] = cv2.addWeighted(roi, 0.80, tint, 0.20, 0)

        annotated = self._draw_hud(annotated, alarm, confidence)
        if self.debug_view:
            dbg = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
            for cand in candidates:
                x, y, bw, bh = cand["bbox"]
                cv2.rectangle(dbg, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
                cv2.putText(
                    dbg, f"{cand.get('score', 0.0) * 100:.0f}%", (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA,
                )
            self.debug_image = dbg
        else:
            self.debug_image = None
        return annotated, alarm

    # ------------------------------------------------------------ persistence
    def _update_persistence(self, fire_detected: bool, now: float) -> bool:
        if fire_detected:
            self._fire_last_seen = now
            if self._fire_first_seen is None:
                self._fire_first_seen = now
            if (now - self._fire_first_seen) >= self._cfg.persistence_threshold:
                self._alarm_active = True
        else:
            if (now - self._fire_last_seen) > self._cfg.persistence_grace:
                self._fire_first_seen = None
            cleared = (
                self._last_score < self._cfg.clear_score
                or self._last_flicker < self._cfg.min_flicker * 0.5
            )
            if cleared and (now - self._fire_last_seen) > self._cfg.clear_delay:
                self._alarm_active = False
        return self._alarm_active

    # -------------------------------------------------------------------- HUD
    def _draw_hud(self, frame: np.ndarray, alarm: bool, confidence: float) -> np.ndarray:
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
        cv2.putText(
            frame, f"conf {confidence * 100:3.0f}%", (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA,
        )
        credit = "Pratik Mishra  |  v5.4"
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
                frame, f"FPS: {avg_fps:.1f}", (16, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA,
            )
        return frame


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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)
        audio_ok = self._audio.initialize(self._cfg)
        if not audio_ok:
            logging.warning("No audio — visual alarm only.")
        self._running = True
        logging.info(f"{self._cfg.window_title} started. Press 'q' to quit, 'd' for masks.")
        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Frame read failed — retrying…")
                    time.sleep(0.05)
                    continue
                frame = cv2.flip(frame, 1)
                annotated, alarm = self._detector.process(frame)
                if alarm and not self._prev_alarm:
                    self._audio.alarm_on()
                    logging.warning("FIRE DETECTED — alarm on")
                elif not alarm and self._prev_alarm:
                    self._audio.alarm_off()
                    logging.info("Fire cleared — alarm off")
                self._prev_alarm = alarm
                self._recorder.notify(alarm, annotated)
                cv2.imshow(self._cfg.window_title, annotated)
                if self._detector.debug_image is not None:
                    cv2.imshow("Fire Masks", self._detector.debug_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("d"):
                    self._detector.debug_view = not self._detector.debug_view
                    if not self._detector.debug_view:
                        try:
                            cv2.destroyWindow("Fire Masks")
                        except cv2.error:
                            pass
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
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Fire Detection System v5.4  |  Pratik Mishra",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", help="Path to JSON config file")
    parser.add_argument("--camera", "-cam", type=int, default=None, help="Camera index")
    parser.add_argument("--debug", action="store_true", help="Start with the mask view enabled")
    args = parser.parse_args()
    if args.config and os.path.exists(args.config):
        config = FireDetectionConfig.from_json(args.config)
        logging.info(f"Config loaded from {args.config}")
    else:
        if args.config:
            logging.warning(f"Config not found: {args.config} — using defaults")
        config = FireDetectionConfig()
    if args.camera is not None:
        config.camera_index = args.camera
    if args.debug:
        config.show_masks = True
    config.validate()
    system = FireDetectionSystem(config)
    system.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())