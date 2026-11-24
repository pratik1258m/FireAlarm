"""
Audio management and siren synthesis for Fire Detection System.
"""
from __future__ import annotations

import logging
import math
import os
import struct
import threading
import wave
from queue import Empty, Queue
from typing import Optional

from src.config import FireDetectionConfig

try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("pygame not installed — audio alarm disabled. Install with: pip install pygame")


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
