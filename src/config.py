"""
Configuration module for Fire Detection System.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Tuple, Union


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

    # --- motion & environmental filtering ---
    use_bg_subtraction: bool = True       # restrict candidates to moving regions only
    bg_history: int = 300                 # MOG2 background model history (frames)
    bg_var_threshold: float = 32.0        # MOG2 pixel variance threshold
    motion_min_overlap: float = 0.18      # fraction of blob pixels that must be in motion
    use_optical_flow: bool = True         # score per-blob edge frame-diff flicker (fast)
    global_light_filter: bool = True      # reject sudden whole-room lighting changes (switches/headlights)
    max_scene_luma_jump: float = 35.0     # max mean luma jump before momentary freeze

    # --- audio ---
    alarm_file: str = "alarm.mp3"
    fallback_wav: str = "alarm_generated.wav"
    alarm_volume: float = 0.85

    # --- WhatsApp (OpenWA Gateway) ---
    enable_whatsapp: bool = True
    openwa_url: str = "http://localhost:8080"
    openwa_session: str = "default"
    openwa_api_key: str = ""
    whatsapp_recipient: str = "918080171693"
    whatsapp_send_snapshot: bool = True
    whatsapp_cooldown: float = 60.0

    # --- runtime & camera ---
    camera_index: Union[int, str] = 0     # 0, 1 for webcams, or 'rtsp://...', 'http://...', 'video.mp4'
    reconnect_interval: float = 2.0       # seconds before retrying disconnected IP cameras
    camera_watchdog_timeout: float = 8.0  # seconds before raising camera offline alert
    resolution: Tuple[int, int] = (1280, 720)
    process_width: int = 640              # downscale width for 25-30 FPS
    fps: int = 30
    window_title: str = "Fire Detection System v5.5  |  Pratik Mishra"
    show_fps: bool = True
    show_masks: bool = False
    headless: bool = False                # run without display window for servers / IoT
    record_on_detection: bool = False
    recording_grace_seconds: int = 10
    output_dir: str = "fire_recordings"
    logs_dir: str = "logs"                # audit logs (CSV and JSON)

    def validate(self) -> None:
        assert self.min_contour_area >= 10, "min_contour_area must be >= 10"
        assert 0 <= self.min_area_ratio <= 1, "min_area_ratio must be 0-1"
        assert 0 < self.max_area_ratio <= 1, "max_area_ratio must be 0-1"
        assert 0 <= self.persistence_threshold, "persistence_threshold must be >= 0"
        assert 0 <= self.alarm_volume <= 1, "alarm_volume must be 0-1"
        assert 0 < self.min_fire_score <= 1, "min_fire_score must be 0-1"
        assert 0 <= self.clear_score <= self.min_fire_score, "clear_score must be <= min_fire_score"
        if self.enable_whatsapp and self.whatsapp_recipient.strip():
            assert self.openwa_url.startswith(("http://", "https://")), "openwa_url must start with http:// or https://"

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
