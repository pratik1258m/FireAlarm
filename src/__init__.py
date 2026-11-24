"""
Fire Alarm System package.
"""
from src.audio import AudioManager
from src.camera import CameraStream
from src.config import FireDetectionConfig
from src.detector import FireDetector
from src.logger import IncidentLogger
from src.notifier import WhatsAppNotifier
from src.recorder import VideoRecorder
from src.system import FireDetectionSystem
from src.utils import _clamp

__all__ = [
    "FireDetectionConfig",
    "AudioManager",
    "CameraStream",
    "IncidentLogger",
    "VideoRecorder",
    "WhatsAppNotifier",
    "FireDetector",
    "FireDetectionSystem",
    "_clamp",
]
