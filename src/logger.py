"""
Regulatory audit trail & incident logging for Fire Detection System.
Maintains persistent compliance logs in CSV and JSON formats.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class IncidentLogger:
    """Logs fire detection events for audit and compliance."""

    def __init__(self, log_dir: str = "logs") -> None:
        self.log_dir = log_dir
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "incidents.csv")
        self.json_path = os.path.join(self.log_dir, "incidents.json")
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        if not os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Incident_ID",
                        "Camera_Source",
                        "Start_Time",
                        "End_Time",
                        "Duration_Sec",
                        "Peak_Confidence_Pct",
                        "Snapshot_Path",
                        "Video_Path",
                        "Status",
                    ])
            except Exception as e:
                logging.error(f"Cannot initialize CSV incident log: {e}")

    def log_incident(
        self,
        camera_source: str,
        start_time: float,
        end_time: float,
        peak_confidence: float,
        snapshot_path: Optional[str] = None,
        video_path: Optional[str] = None,
        status: str = "RESOLVED",
    ) -> Dict[str, Any]:
        duration = round(max(0.0, end_time - start_time), 2)
        start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
        incident_id = f"FIRE-{int(start_time)}"

        record = {
            "incident_id": incident_id,
            "camera_source": str(camera_source),
            "start_time": start_str,
            "end_time": end_str,
            "duration_sec": duration,
            "peak_confidence_pct": round(peak_confidence * 100, 1),
            "snapshot_path": snapshot_path or "N/A",
            "video_path": video_path or "N/A",
            "status": status,
        }

        # 1. Append to CSV
        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    record["incident_id"],
                    record["camera_source"],
                    record["start_time"],
                    record["end_time"],
                    record["duration_sec"],
                    record["peak_confidence_pct"],
                    record["snapshot_path"],
                    record["video_path"],
                    record["status"],
                ])
        except Exception as e:
            logging.error(f"Failed to append to CSV incident log: {e}")

        # 2. Append to JSON array
        try:
            existing = []
            if os.path.exists(self.json_path):
                try:
                    with open(self.json_path, "r") as f:
                        existing = json.load(f)
                except Exception:
                    existing = []
            existing.append(record)
            with open(self.json_path, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to update JSON incident log: {e}")

        logging.info(f"Audit log created: {incident_id} (Duration: {duration}s, Peak: {record['peak_confidence_pct']}%)")
        return record
