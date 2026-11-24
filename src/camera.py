"""
Robust, low-latency camera stream handler supporting Webcams, RTSP CCTV streams,
HTTP mobile camera streams, and video files with automatic reconnection.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np


class CameraStream:
    """
    Threaded camera stream reader that eliminates RTSP buffering lag
    and provides automatic reconnection on network drops.
    """

    def __init__(
        self,
        source: Union[int, str],
        resolution: Tuple[int, int] = (1280, 720),
        target_fps: int = 30,
        reconnect_interval: float = 2.0,
    ) -> None:
        self.source = source
        self.resolution = resolution
        self.target_fps = target_fps
        self.reconnect_interval = reconnect_interval

        self._is_file = isinstance(source, str) and not source.startswith(("rtsp://", "http://", "https://"))
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._last_frame_time = 0.0

    def start(self) -> bool:
        if not self._open_capture():
            if not self._is_file:
                logging.warning(
                    f"Initial connection to {self.source} failed; auto-reconnect will retry in background."
                )
            else:
                logging.error(f"Cannot open video file: {self.source}")
                return False

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._capture_worker, daemon=True, name="CameraReader")
        self._thread.start()
        return True

    def _open_capture(self) -> bool:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass

        if isinstance(self.source, str) and self.source.isdigit():
            src: Union[int, str] = int(self.source)
        else:
            src = self.source

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self._connected = False
            return False

        if not self._is_file:
            # Set buffer size to 1 to eliminate RTSP/CCTV delay
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._cap = cap
        self._connected = True
        return True

    def _capture_worker(self) -> None:
        while not self._stop_evt.is_set():
            if not self._connected or self._cap is None or not self._cap.isOpened():
                if self._is_file:
                    break
                time.sleep(self.reconnect_interval)
                logging.info(f"Attempting camera reconnection to: {self.source}...")
                self._open_capture()
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                if self._is_file:
                    logging.info("Video file playback finished.")
                    break
                logging.warning(f"Frame read dropped from {self.source}; reconnecting...")
                self._connected = False
                time.sleep(0.5)
                continue

            with self._lock:
                self._frame = frame
                self._last_frame_time = time.time()

            # Small yield to prevent CPU pegging
            time.sleep(0.002)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def is_opened(self) -> bool:
        return self._connected or (self._thread is not None and self._thread.is_alive())

    def release(self) -> None:
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._connected = False
