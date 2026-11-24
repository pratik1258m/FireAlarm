# 🔥 Fire Detection Alarm System v5.4

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue?style=for-the-badge&logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A high-performance, real-time fire detection system utilizing research-backed multi-color-space analysis, MOG2 motion gating, fast frame-diff flicker verification, and hysteresis persistence to eliminate false triggers. 🚨**

</div>

---

## Features

- **Multi-Color-Space Pipeline**:
  - **Chen et al. (2004) RGB rule** ($R \ge G > B$ with $G - B$ gap) to reject red cloth, pillows, and clothing.
  - **Celik & Demirel (2009) YCbCr model** adaptive to room lighting to reject bright beige walls.
  - **White-hot core + fire halo validation** to reliably catch small, overexposed match and lighter flames.
- **Motion Gating & Fast Flicker Tracking**:
  - MOG2 background subtraction ensures static warm objects are ignored.
  - High-speed per-blob frame-differencing flicker analysis maintaining a smooth **25–30 FPS**.
- **Temporal Persistence Guard**:
  - Hysteresis confidence gate prevents false triggers and eliminates stuck alarms.
- **Thread-Safe Audio Alarm**:
  - Plays dedicated alarm sound (`alarm.mp3`) with an automated synthetic siren generator fallback.
- **HUD & Debug View**:
  - Minimal real-time HUD with FPS and confidence score; press **`d`** to toggle mask and candidate debugging.
- **Optional Incident Recording**:
  - Automatically records and saves video clips upon fire detection.

---

## Requirements

Python 3.8+ and dependencies:

```bash
pip install -r requirements.txt
```

*(Optional for audio)*:
```bash
pip install pygame
```

---

## Usage

Run with default camera:
```bash
python3 main.py
```

Optional CLI flags:
```bash
python3 main.py --camera 1              # Select camera index
python3 main.py --debug                 # Launch with debug masks enabled
python3 main.py --config config.json    # Load custom configuration
```

- Press **`q`** to quit.
- Press **`d`** to toggle live mask and candidate debug visualization.

---

## Detection Pipeline & Weights

Candidate blobs are scored on multi-factor evidence:

| Feature | Weight | What it proves |
|---|---|---|
| **Core Light** | 26% | Localized bright core inside flame body (rejects flat orange/red items) |
| **Chrominance** | 22% | $Cr - Cb$ chrominance strength under adaptive ambient lighting |
| **Temporal Flicker** | 30% | Area oscillation, silhouette IoU fluctuation, and luminance change |
| **Frame-Diff Flicker** | 14% | Micro-level pixel difference across consecutive frames |
| **Internal Variation** | 8% | Gradients and texture differences within flame boundary |

---

## Tuning (in `FireDetectionConfig`)

| Option | Default | Purpose |
|---|---|---|
| `min_fire_score` | `0.65` | Confidence required to confirm fire and trigger alarm. |
| `clear_score` | `0.52` | Confidence threshold below which suspect boxes clear. |
| `min_flicker` | `0.22` | Minimum temporal flicker required (static objects rejected). |
| `warmup_frames` | `8` | Number of frames of history required before temporal score applies. |
| `persistence_threshold` | `1.5` | Seconds of sustained detection before audio alarm fires. |
| `clear_delay` | `3.5` | Seconds of clear feed before alarm stops. |
| `min_contour_area` | `40` | Minimum candidate contour area in pixels. |
| `use_bg_subtraction` | `True` | Gate detection to active motion areas via MOG2. |

---

## License

MIT License — © 2026 Pratik Mishra. See [LICENSE](LICENSE).
