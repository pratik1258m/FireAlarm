"""
Video recording module for Fire Detection System.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import FireDetectionConfig


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
