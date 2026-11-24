"""
WhatsApp notification module for Fire Detection System using OpenWA REST server.
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from queue import Empty, Queue
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import FireDetectionConfig


class WhatsAppNotifier:
    """
    Thread-safe, asynchronous WhatsApp alert notifier using OpenWA REST server.
    Dispatches alerts with photos without blocking the video capture/processing loop.
    """

    def __init__(self, config: FireDetectionConfig) -> None:
        self._cfg = config
        self._enabled = config.enable_whatsapp and bool(config.whatsapp_recipient.strip())
        self._recipient = self._normalize_recipient(config.whatsapp_recipient)
        self._queue: Queue[dict] = Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._last_alert_time: float = 0.0

        if self._enabled:
            self._start_thread()
            logging.info(f"WhatsApp alerts enabled for {self._recipient} via OpenWA ({config.openwa_url})")
        elif config.enable_whatsapp:
            logging.warning("WhatsApp enabled but recipient is empty. Please set --wa-recipient or update config.")

    @staticmethod
    def _normalize_recipient(recipient: str) -> str:
        r = recipient.strip()
        if not r:
            return ""
        if "@" in r:
            return r
        digits = "".join(ch for ch in r if ch.isdigit())
        if len(digits) == 10:
            digits = f"91{digits}"
        return f"{digits}@c.us" if digits else r

    def _start_thread(self) -> None:
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="WhatsAppNotifier")
        self._thread.start()

    def _worker_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                task = self._queue.get(timeout=0.25)
                self._dispatch(task)
                self._queue.task_done()
            except Empty:
                pass
            except Exception as e:
                logging.error(f"WhatsApp worker loop error: {e}")

    def _dispatch(self, task: dict) -> None:
        ttype = task.get("type")
        if ttype == "fire_alert":
            self._send_fire_alert(
                task.get("confidence", 0.0),
                task.get("frame"),
                task.get("timestamp", time.time()),
            )
        elif ttype == "fire_cleared":
            self._send_fire_cleared(
                task.get("duration", 0.0),
                task.get("timestamp", time.time()),
            )
        elif ttype == "camera_offline":
            self._send_camera_offline(
                task.get("camera_source", "Unknown"),
                task.get("timestamp", time.time()),
            )
        elif ttype == "camera_restored":
            self._send_camera_restored(
                task.get("camera_source", "Unknown"),
                task.get("timestamp", time.time()),
            )

    def _post(self, url: str, data: dict, timeout: float = 8.0) -> Tuple[bool, int, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "FireAlarm-System/5.4",
        }
        if self._cfg.openwa_api_key:
            headers["x-api-key"] = self._cfg.openwa_api_key
            headers["Authorization"] = f"Bearer {self._cfg.openwa_api_key}"

        payload_bytes = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                body = resp.read().decode("utf-8", errors="replace")
                return True, status, body
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
            return False, e.code, err_body
        except urllib.error.URLError as e:
            return False, 0, f"Connection failed: {e.reason}"
        except Exception as e:
            return False, 0, str(e)

    def _send_text(self, text: str) -> bool:
        base_url = self._cfg.openwa_url.rstrip("/")

        # Strategy 1: OpenWA Standard /api/messages/sendText
        url_std = f"{base_url}/api/messages/sendText"
        body_std = {"to": self._recipient, "content": text}
        ok, code, res = self._post(url_std, body_std)
        if ok and code in (200, 201):
            return True

        # Strategy 2: Session endpoint (/api/sessions/{session}/messages/send-text)
        url_session = f"{base_url}/api/sessions/{self._cfg.openwa_session}/messages/send-text"
        body_session = {"chatId": self._recipient, "text": text}
        ok, code, res = self._post(url_session, body_session)
        if ok and code in (200, 201):
            return True

        # Strategy 3: Direct /sendText with args wrapper
        url_direct = f"{base_url}/sendText"
        body_direct = {"args": {"to": self._recipient, "content": text}}
        ok, code, res = self._post(url_direct, body_direct)
        if ok and code in (200, 201):
            return True

        logging.error(f"WhatsApp notification delivery failed: {res}")
        return False

    def _send_image(self, frame: np.ndarray, caption: str) -> bool:
        base_url = self._cfg.openwa_url.rstrip("/")
        try:
            ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ret:
                return self._send_text(caption)
            b64_data = base64.b64encode(buf).decode("utf-8")
        except Exception as e:
            logging.error(f"JPEG encoding failed: {e}")
            return self._send_text(caption)

        # Strategy 1: OpenWA Standard /api/messages/sendImage
        url_std = f"{base_url}/api/messages/sendImage"
        body_std = {
            "to": self._recipient,
            "file": f"data:image/jpeg;base64,{b64_data}",
            "filename": "fire_snapshot.jpg",
            "options": {"caption": caption}
        }
        ok, code, res = self._post(url_std, body_std, timeout=15.0)
        if ok and code in (200, 201):
            return True

        # Strategy 2: Session endpoint with file object
        url_session = f"{base_url}/api/sessions/{self._cfg.openwa_session}/messages/send-image"
        body_session_file = {
            "chatId": self._recipient,
            "file": {
                "mimetype": "image/jpeg",
                "data": f"data:image/jpeg;base64,{b64_data}",
                "filename": "fire_snapshot.jpg",
            },
            "caption": caption,
        }
        ok, code, res = self._post(url_session, body_session_file, timeout=12.0)
        if ok and code in (200, 201):
            return True

        # Strategy 3: Direct /sendImage with args
        url_direct = f"{base_url}/sendImage"
        body_direct = {
            "args": {
                "to": self._recipient,
                "file": f"data:image/jpeg;base64,{b64_data}",
                "filename": "fire_snapshot.jpg",
                "caption": caption,
            }
        }
        ok, code, res = self._post(url_direct, body_direct, timeout=12.0)
        if ok and code in (200, 201):
            return True

        # Fallback to plain text if image dispatch rejected
        logging.warning("WhatsApp image dispatch failed, falling back to text notification.")
        return self._send_text(caption)

    def _send_fire_alert(self, confidence: float, frame: Optional[np.ndarray], ts: float) -> None:
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        conf_pct = confidence * 100
        caption = (
            "🚨 *CRITICAL EMERGENCY: FIRE DETECTED!* 🚨\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 *Location:* Camera {self._cfg.camera_index}\n"
            f"🔥 *Status:* ACTIVE FLAME CONFIRMED\n"
            f"📊 *Confidence:* {conf_pct:.1f}%\n"
            f"⏱ *Timestamp:* {dt}\n"
            "\n"
            "⚠️ *DETAILS:* Multi-spectrum signature verified (luminance core, YCbCr chroma, flicker & motion).\n"
            "🏃 *ACTION:* Evacuate immediately and verify area safety!\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "_Fire Detection System v5.4 | Pratik Mishra_"
        )
        logging.info(f"Dispatching WhatsApp fire alert to {self._recipient}")
        if frame is not None and self._cfg.whatsapp_send_snapshot:
            self._send_image(frame, caption)
        else:
            self._send_text(caption)

    def _send_fire_cleared(self, duration: float, ts: float) -> None:
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        mins, secs = divmod(int(duration), 60)
        dur_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        msg = (
            "✅ *FIRE INCIDENT RESOLVED* ✅\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 *Location:* Camera {self._cfg.camera_index}\n"
            f"⏱ *Cleared At:* {dt}\n"
            f"⏳ *Incident Duration:* {dur_str}\n"
            "🟢 *Status:* Area clear — continuous monitoring resumed\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "_Fire Detection System v5.4 | Pratik Mishra_"
        )
        logging.info(f"Dispatching WhatsApp fire cleared notification to {self._recipient}")
        self._send_text(msg)

    def _send_camera_offline(self, camera_source: str, ts: float) -> None:
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        msg = (
            "⚠️ *CRITICAL SYSTEM WARNING: CAMERA OFFLINE!* ⚠️\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 *Camera:* {camera_source}\n"
            f"⏱ *Timestamp:* {dt}\n"
            "\n"
            "⚠️ *STATUS:* Video stream has timed out or disconnected! Sector fire detection is currently UNGUARDED.\n"
            "🔧 *ACTION:* Inspect camera power supply, Wi-Fi, or cable connection immediately.\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "_Fire Detection System v5.5 | Pratik Mishra_"
        )
        logging.warning(f"Dispatching camera offline alert to {self._recipient}")
        self._send_text(msg)

    def _send_camera_restored(self, camera_source: str, ts: float) -> None:
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        msg = (
            "🟢 *CAMERA FEED RESTORED* 🟢\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 *Camera:* {camera_source}\n"
            f"⏱ *Timestamp:* {dt}\n"
            "✅ *STATUS:* Connection successfully re-established. Active fire detection online.\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "_Fire Detection System v5.5 | Pratik Mishra_"
        )
        logging.info(f"Dispatching camera restored alert to {self._recipient}")
        self._send_text(msg)

    def notify_camera_offline(self, camera_source: str) -> None:
        if not self._enabled:
            return
        self._queue.put({
            "type": "camera_offline",
            "camera_source": str(camera_source),
            "timestamp": time.time(),
        })

    def notify_camera_restored(self, camera_source: str) -> None:
        if not self._enabled:
            return
        self._queue.put({
            "type": "camera_restored",
            "camera_source": str(camera_source),
            "timestamp": time.time(),
        })

    def notify_fire(self, confidence: float, frame: Optional[np.ndarray]) -> None:
        if not self._enabled:
            return
        now = time.time()
        if self._last_alert_time > 0 and (now - self._last_alert_time) < self._cfg.whatsapp_cooldown:
            return
        self._last_alert_time = now
        frame_copy = frame.copy() if frame is not None else None
        self._queue.put({
            "type": "fire_alert",
            "confidence": confidence,
            "frame": frame_copy,
            "timestamp": now,
        })

    def notify_cleared(self, duration: float) -> None:
        if not self._enabled:
            return
        now = time.time()
        self._last_alert_time = 0.0
        self._queue.put({
            "type": "fire_cleared",
            "duration": duration,
            "timestamp": now,
        })

    def send_test_sync(self) -> bool:
        if not self._enabled:
            logging.error("WhatsApp is not configured. Provide --wa-recipient to send test.")
            return False
        dt = time.strftime("%Y-%m-%d %H:%M:%S")
        test_msg = (
            "🔔 *FIRE ALARM SYSTEM — WHATSAPP TEST* 🔔\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "✅ *OpenWA Gateway Connection:* ACTIVE & VERIFIED\n"
            f"📍 *Camera Index:* {self._cfg.camera_index}\n"
            f"⏱ *Timestamp:* {dt}\n"
            "\n"
            "ℹ️ *System Status:* Ready to dispatch instant fire alerts with visual snapshots.\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "_Fire Detection System v5.4 | Pratik Mishra_"
        )
        logging.info(f"Sending test WhatsApp message to {self._recipient} via {self._cfg.openwa_url}...")
        ok = self._send_text(test_msg)
        if ok:
            logging.info("✅ Test WhatsApp message delivered successfully!")
        else:
            logging.error(f"❌ Failed to deliver test message to OpenWA server ({self._cfg.openwa_url}).")
            logging.error("   Ensure your OpenWA server is running (e.g. npx @open-wa/wa-automate --port 8080) and scanned.")
        return ok

    def shutdown(self) -> None:
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
