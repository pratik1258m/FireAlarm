# 🔥 Fire Detection & Safety Automation System v5.5

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-yellow?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue?style=for-the-badge&logo=opencv&logoColor=white)
![WhatsApp](https://img.shields.io/badge/WhatsApp-OpenWA-25D366?style=for-the-badge&logo=whatsapp&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-Modular%20v5.5-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An industrial-grade, modular, real-time fire and flame detection platform combining research-backed dual-spectrum computer vision, zero-lag IP/mobile camera streaming, a cyberpunk telemetry HUD, black-box forensic logging, and automated WhatsApp emergency dispatch with high-res photo snapshots.** 🚨

</div>

---

## 📑 Table of Contents
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Detection Pipeline & Research Algorithms](#-detection-pipeline--research-algorithms)
- [Quick Start & Installation](#-quick-start--installation)
- [Mobile Camera Integration (IP Webcam)](#-mobile-camera-integration-ip-webcam)
- [WhatsApp Emergency Alerts (OpenWA)](#-whatsapp-emergency-alerts-openwa)
- [Command-Line Reference](#-command-line-reference)
- [Forensic Incident Logging](#-forensic-incident-logging)
- [Configuration Reference](#-configuration-reference)
- [License & Author](#-license--author)

---

## 🏛 System Architecture

The project has been refactored from a monolithic script into an **8-module, thread-isolated architecture** located in [`src/`](file:///Users/pratik/Documents/FireAlarm/src/):

```
FireAlarm/
├── main.py                     # Lean CLI entrypoint & argument parser
├── src/
│   ├── __init__.py             # Package exports & version metadata (v5.5)
│   ├── config.py               # Typed dataclass config with validation
│   ├── camera.py               # Asynchronous zero-lag threaded camera stream
│   ├── detector.py             # Dual-color CV pipeline & Cyberpunk HUD renderer
│   ├── notifier.py             # Thread-safe WhatsApp dispatcher (OpenWA REST)
│   ├── logger.py               # Black-box forensic incident logger (CSV & JSON)
│   ├── audio.py                # Pygame MP3 player + synthesized sine siren fallback
│   ├── recorder.py             # Circular ring buffer MP4 evidence video recorder
│   └── system.py               # Master orchestrator coordinating all subsystems
├── logs/
│   ├── incidents.csv           # Tabular incident audit trail
│   └── incidents.json          # Machine-readable incident log
├── captures/                   # High-res incident snapshot captures
├── recordings/                 # Evidence video clips (MP4)
└── requirements.txt            # Python dependencies
```

---

## ✨ Key Features

- **🔬 Dual-Spectrum Flame Segmentation**:
  - **Chen et al. (2004) RGB rule**: Strict $R \ge G > B$ and $G - B$ chrominance gap filter rejects red clothing, fabrics, and toys.
  - **Celik & Demirel (2009) YCbCr model**: Adaptive lighting thresholding prevents false alarms from warm incandescent bulbs and beige walls.
  - **Overexposed Core & Halo Extractor**: Detects tiny white-hot flame cores (e.g., matchsticks, lighters) surrounded by luminous flame boundaries.
- **⚡ Zero-Lag Asynchronous Streaming**:
  - Independent background acquisition thread with `cv2.CAP_PROP_BUFFERSIZE = 1` ring buffer eliminates network buffering latency on wireless/mobile cameras.
  - Automatic reconnection watchdog with exponential retry intervals for dropped RTSP or Wi-Fi feeds.
- **📱 Mobile & CCTV Camera Support**:
  - Seamlessly streams from built-in webcams, Android/iOS IP Webcam apps (`http://.../video`), CCTV RTSP streams (`rtsp://...`), or pre-recorded MP4 files.
- **💬 Automated WhatsApp Emergency Dispatch**:
  - Dispatches emergency notifications to personal numbers or group chats.
  - Sends full-resolution annotated camera snapshots showing flame bounding boxes, confidence score, and timestamp.
  - Automatic phone number normalization (supports 10-digit Indian numbers `8080171693` $\rightarrow$ `918080171693@c.us`).
  - Cooldown suppression avoids notification flooding during continuous fires.
  - Sends a "Camera Offline" alert if the video feed drops and a "Camera Restored" alert once it reconnects.
- **🎨 Futuristic Cyberpunk HUD**:
  - Frosted glass telemetry bars, corner brackets, real-time FPS counter, confidence meter, and pulsing high-contrast bounding boxes.
  - Live candidate debug mask view toggled with a single keystroke (`d`).
- **🗄 Black-Box Forensic Incident Logging**:
  - Automatic audit trail saved to [`logs/incidents.csv`](file:///Users/pratik/Documents/FireAlarm/logs/incidents.csv) and `logs/incidents.json` recording incident ID, start time, end time, duration, peak confidence, and snapshot paths.
- **🔊 Dual-Engine Audio Alarm**:
  - Pygame MP3 playback (`alarm.mp3`) with an automated fallback to an algorithmic sine-wave frequency-sweep siren generator.
- **🛡 Headless / IoT Server Mode**:
  - Run completely windowless (`--headless`) for low-power Raspberry Pi, Linux servers, and Docker containers.

---

## 🔬 Detection Pipeline & Research Algorithms

Each video frame passes through a multi-stage computer vision pipeline:

```
[Camera Stream]
       │
       ▼
[Downscale to 640px]
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[Chen RGB Segmentation]         [Celik YCbCr Segmentation]
       │                                 │
       └──────────────┬──────────────────┘
                      ▼
            [Intersection Mask]
                      │
                      ▼
         [MOG2 Background Motion Gate]
                      │
                      ▼
         [Morphological Cleanup (Close/Open)]
                      │
                      ▼
            [Candidate Blob Analysis]
                      │
       ┌──────────────┴──────────────┐
       ▼                             ▼
[Temporal Flicker Tracker]    [White-Hot Core Verifier]
       │                             │
       └──────────────┬──────────────┘
                      ▼
       [Hysteresis Persistence Filter]
                      │
          Score > min_fire_score ?
             ├── YES ──► [TRIGGER ALARM, WHATSAPP, AUDIO, LOG, RECORD]
             └── NO  ──► [MONITORING STANDBY]
```

### Evidence Scoring Weights:
| Evidence Factor | Weight | Purpose |
|---|:---:|---|
| **Temporal Flicker** | **30%** | Flame perimeter oscillation, area variance, and centroid movement (rejects static warm objects) |
| **White-Hot Core** | **26%** | High-intensity luminance core inside flame boundary (differentiates real fire from red cloths) |
| **YCbCr Chrominance** | **22%** | $Cr - Cb$ color contrast calibrated against adaptive scene illumination |
| **Edge Frame-Diff** | **14%** | Fast micro-level pixel fluctuations across consecutive frames |
| **Texture Gradient** | **8%** | Internal entropy and gradient variation within flame contour |

---

## 🚀 Quick Start & Installation

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/pratik-mishra/FireAlarm.git
cd FireAlarm

# Install Python requirements
pip install -r requirements.txt
pip install pygame
```

### 2. Run with Laptop Webcam
```bash
python3 main.py
```

### 3. Interactive Shortcuts
- **`q`** : Cleanly stops the system and closes all streams.
- **`d`** : Toggle between the Cyberpunk HUD view and the CV Debug Mask view.

---

## 📱 Mobile Camera Integration (IP Webcam)

Turn any Android or iOS smartphone into a wireless fire detection camera:

1. Install **IP Webcam** (Android) or **IP Camera Lite** (iOS) from the app store.
2. Connect your phone and computer to the **same Wi-Fi network**.
3. Open the app and tap **Start Server** at the bottom.
4. Note the URL shown on your phone screen (e.g., `http://192.0.0.4:8080`).
5. Run the Fire Alarm pointing to the stream URL:
   ```bash
   python3 main.py --camera http://192.0.0.4:8080/video
   ```

The zero-lag capture engine automatically sets `CAP_PROP_BUFFERSIZE = 1` and discards stale frames, giving you real-time **30–40 FPS** tracking over Wi-Fi!

---

## 💬 WhatsApp Emergency Alerts (OpenWA)

The system integrates directly with an OpenWA REST API server running on your network.

### 1. Launch OpenWA Gateway
In a terminal window, start the gateway:
```bash
npx -y @open-wa/wa-automate --port 8080
```

### 2. Link Your WhatsApp Account
- Open your browser to: **`http://localhost:8080/dashboard/`**
- Or scan the terminal QR code using WhatsApp on your phone:
  - Go to **WhatsApp** $\rightarrow$ **Settings** (or 3 dots) $\rightarrow$ **Linked Devices** $\rightarrow$ **Link a Device**.
- Once authenticated, the gateway status switches to `READY`.

### 3. Run with WhatsApp Alerts Enabled
```bash
# Using 10-digit Indian phone number
python3 main.py --camera http://192.0.0.4:8080/video --whatsapp --wa-recipient 8080171693

# Using international phone number
python3 main.py --whatsapp --wa-recipient 918080171693

# Using WhatsApp Group ID
python3 main.py --whatsapp --wa-recipient 120363028192839182@g.us
```

### 4. Send a WhatsApp Startup Self-Test
Verify that your WhatsApp connection works before monitoring:
```bash
python3 main.py --wa-test --wa-recipient 8080171693
```

---

## 💻 Command-Line Reference

```bash
python3 main.py [OPTIONS]
```

| Option | Shorthand | Default | Description |
|---|:---:|:---:|---|
| `--camera` | `-cam` | `0` | Camera index (`0`, `1`) or URL (`http://.../video`, `rtsp://...`, `video.mp4`) |
| `--whatsapp` | `-wa` | `True` | Enable automated WhatsApp emergency notifications |
| `--wa-recipient` | `-war` | `8080171693` | Recipient WhatsApp phone number or Group JID |
| `--wa-url` | — | `http://localhost:8080` | OpenWA REST Server URL |
| `--wa-session` | — | `default` | OpenWA session identifier |
| `--wa-key` | — | `""` | OpenWA API Key (optional) |
| `--wa-no-snapshot` | — | `False` | Send text-only WhatsApp alerts (suppresses photo snapshots) |
| `--wa-test` | — | `False` | Send an instant test alert on startup and exit |
| `--debug` | — | `False` | Start with CV candidate debug mask enabled |
| `--headless` | — | `False` | Run windowless for servers, IoT, and background daemons |
| `--config` | `-c` | `None` | Path to optional JSON configuration file |

---

## 🗄 Forensic Incident Logging

Every fire event is treated as an official safety incident and saved with forensic metadata.

### 1. CSV Audit Trail (`logs/incidents.csv`)
```csv
incident_id,start_time,end_time,duration_seconds,max_confidence,max_flame_area_px,camera_index,snapshot_path,recording_path
INC-20261124-140500,2026-11-24 14:05:00,2026-11-24 14:05:14,14.2,0.942,1850,http://192.0.0.4:8080/video,captures/INC-20261124-140500.jpg,recordings/INC-20261124-140500.mp4
```

### 2. Machine-Readable JSON (`logs/incidents.json`)
Structured for SIEM / SOC dashboard ingestion, analytics, or webhook triggers.

---

## ⚙ Configuration Reference

All settings can be customized via [`src/config.py`](file:///Users/pratik/Documents/FireAlarm/src/config.py) or loaded via `--config custom_config.json`:

```json
{
  "min_fire_score": 0.65,
  "clear_score": 0.52,
  "min_flicker": 0.22,
  "warmup_frames": 8,
  "persistence_threshold": 1.5,
  "clear_delay": 3.5,
  "min_contour_area": 40,
  "use_bg_subtraction": true,
  "enable_whatsapp": true,
  "whatsapp_recipient": "918080171693",
  "whatsapp_send_snapshot": true,
  "whatsapp_cooldown": 60.0,
  "camera_watchdog_timeout": 8.0,
  "headless": false
}
```

---

## 👨‍💻 License & Author

- **Author**: **Pratik Mishra**
- **Project**: Fire Detection & Safety Automation System (v5.5 Modular)
- **License**: MIT License — See [LICENSE](LICENSE) for details.
