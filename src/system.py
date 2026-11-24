"""
System orchestrator wiring detector, audio, video recorder, and WhatsApp notifier.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import cv2

import os
from src.audio import AudioManager
from src.camera import CameraStream
from src.config import FireDetectionConfig
from src.detector import FireDetector
from src.logger import IncidentLogger
from src.notifier import WhatsAppNotifier
from src.recorder import VideoRecorder


class FireDetectionSystem:
    """Wires detector, audio, recorder, watchdog, and audit logger together."""

    def __init__(self, config: FireDetectionConfig) -> None:
        self._cfg = config
        self._detector = FireDetector(config)
        self._audio = AudioManager()
        self._recorder = VideoRecorder(config)
        self._notifier = WhatsAppNotifier(config)
        self._logger = IncidentLogger(config.logs_dir)
        self._camera = CameraStream(
            config.camera_index,
            config.resolution,
            config.fps,
            config.reconnect_interval,
        )
        self._running = False
        self._prev_alarm = False
        self._alarm_start_time: Optional[float] = None
        self._peak_confidence: float = 0.0
        self._last_snapshot_path: Optional[str] = None
        self._last_frame_time: float = time.time()
        self._watchdog_fired: bool = False

    def run(self) -> None:
        if not self._camera.start():
            logging.error(f"Cannot initialize camera source: {self._cfg.camera_index}")
            return

        audio_ok = self._audio.initialize(self._cfg)
        if not audio_ok:
            logging.warning("No audio — visual alarm only.")
        self._running = True
        mode_str = "HEADLESS MODE (no GUI)" if self._cfg.headless else "GUI MODE (press 'q' to quit, 'd' for masks)"
        logging.info(f"{self._cfg.window_title} started in {mode_str}.")

        is_webcam = isinstance(self._cfg.camera_index, int) or (
            isinstance(self._cfg.camera_index, str) and self._cfg.camera_index.isdigit()
        )

        try:
            while self._running:
                ret, frame = self._camera.read()
                now = time.time()

                # Camera watchdog check
                if not ret or frame is None:
                    if (now - self._last_frame_time) > self._cfg.camera_watchdog_timeout:
                        if not self._watchdog_fired:
                            logging.warning(f"Watchdog: Camera stream {self._cfg.camera_index} offline!")
                            self._notifier.notify_camera_offline(str(self._cfg.camera_index))
                            self._watchdog_fired = True
                    time.sleep(0.01)
                    continue

                # Frame received successfully
                self._last_frame_time = now
                if self._watchdog_fired:
                    logging.info(f"Watchdog: Camera stream {self._cfg.camera_index} restored!")
                    self._notifier.notify_camera_restored(str(self._cfg.camera_index))
                    self._watchdog_fired = False

                if is_webcam:
                    frame = cv2.flip(frame, 1)

                annotated, alarm = self._detector.process(frame)

                # Track peak confidence during alarm event
                if alarm:
                    self._peak_confidence = max(self._peak_confidence, self._detector._last_score)

                # State transition: ALARM TRIGGERED
                if alarm and not self._prev_alarm:
                    self._audio.alarm_on()
                    self._alarm_start_time = now
                    self._peak_confidence = self._detector._last_score

                    # Save snapshot image for audit trail
                    ts_str = time.strftime("%Y%m%d_%H%M%S")
                    snap_dir = os.path.join(self._cfg.output_dir, "snapshots")
                    os.makedirs(snap_dir, exist_ok=True)
                    self._last_snapshot_path = os.path.join(snap_dir, f"snapshot_{ts_str}.jpg")
                    cv2.imwrite(self._last_snapshot_path, annotated)

                    self._notifier.notify_fire(confidence=self._detector._last_score, frame=annotated)
                    logging.warning("FIRE DETECTED — alarm on & WhatsApp alert dispatched")

                # State transition: ALARM CLEARED
                elif not alarm and self._prev_alarm:
                    self._audio.alarm_off()
                    dur = now - (self._alarm_start_time or now)
                    self._notifier.notify_cleared(duration=dur)

                    # Log regulatory audit trail
                    self._logger.log_incident(
                        camera_source=str(self._cfg.camera_index),
                        start_time=self._alarm_start_time or (now - dur),
                        end_time=now,
                        peak_confidence=self._peak_confidence,
                        snapshot_path=self._last_snapshot_path,
                        video_path=self._recorder._filepath,
                        status="RESOLVED",
                    )

                    self._alarm_start_time = None
                    self._peak_confidence = 0.0
                    logging.info(f"Fire cleared (duration {dur:.1f}s) — alarm off & incident logged")

                self._prev_alarm = alarm
                self._recorder.notify(alarm, annotated)

                # GUI display (skipped in headless mode)
                if not self._cfg.headless:
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
                else:
                    time.sleep(0.001)

        except KeyboardInterrupt:
            logging.info("Interrupted by user.")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        self._running = False
        self._audio.alarm_off()
        time.sleep(0.15)
        self._audio.shutdown()
        self._recorder.close()
        self._notifier.shutdown()
        self._camera.release()
        if not self._cfg.headless:
            cv2.destroyAllWindows()
        logging.info("System shut down cleanly.")
