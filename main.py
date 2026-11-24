"""
Fire Detection Alarm System v5.4 - Production Ready
Developed by Pratik Mishra

Modular Architecture:
- src.config   : Configuration dataclass & JSON serialization
- src.audio    : Thread-safe audio alarm manager & fallback siren synthesis
- src.recorder : Incident video recording
- src.notifier : Asynchronous WhatsApp alerts via OpenWA with photo snapshots
- src.detector : CV pipeline (Chen RGB, Celik YCbCr, MOG2 motion, frame-diff flicker, tracking)
- src.system   : System orchestrator wiring all components together

Usage:
    python main.py
    python main.py --camera 1
    python main.py --debug
    python main.py --whatsapp --wa-recipient 919876543210
    python main.py --wa-test --wa-recipient 919876543210
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

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
    "main",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Fire Detection System v5.5  |  Pratik Mishra",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", help="Path to JSON config file")
    parser.add_argument(
        "--camera",
        "-cam",
        type=str,
        default=None,
        help="Camera index (0, 1) or RTSP/HTTP URL or video file path",
    )
    parser.add_argument("--debug", action="store_true", help="Start with the mask view enabled")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window for servers/IoT")
    parser.add_argument("--whatsapp", "-wa", action="store_true", help="Enable WhatsApp notifications via OpenWA")
    parser.add_argument("--wa-recipient", "-war", type=str, default=None, help="WhatsApp recipient phone number or group ID (e.g. 919876543210 or 12345@g.us)")
    parser.add_argument("--wa-url", type=str, default=None, help="OpenWA REST server URL (default: http://localhost:8080)")
    parser.add_argument("--wa-session", type=str, default=None, help="OpenWA session ID (default: default)")
    parser.add_argument("--wa-key", type=str, default=None, help="OpenWA API Key (optional)")
    parser.add_argument("--wa-no-snapshot", action="store_true", help="Disable sending image snapshots in WhatsApp alerts")
    parser.add_argument("--wa-test", action="store_true", help="Send a WhatsApp test notification on startup")
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        config = FireDetectionConfig.from_json(args.config)
        logging.info(f"Config loaded from {args.config}")
    else:
        if args.config:
            logging.warning(f"Config not found: {args.config} — using defaults")
        config = FireDetectionConfig()

    if args.camera is not None:
        config.camera_index = int(args.camera) if args.camera.isdigit() else args.camera
    if args.debug:
        config.show_masks = True
    if args.headless:
        config.headless = True
    if args.whatsapp or args.wa_recipient is not None:
        config.enable_whatsapp = True
    if args.wa_recipient is not None:
        config.whatsapp_recipient = args.wa_recipient
    if args.wa_url is not None:
        config.openwa_url = args.wa_url
    if args.wa_session is not None:
        config.openwa_session = args.wa_session
    if args.wa_key is not None:
        config.openwa_api_key = args.wa_key
    if args.wa_no_snapshot:
        config.whatsapp_send_snapshot = False

    config.validate()
    system = FireDetectionSystem(config)

    if args.wa_test:
        system._notifier.send_test_sync()

    system.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
