# 🔥 Fire Detection Alarm System v2.0

A real-time fire detection system using OpenCV and HSV colour masking.  
Developed by **Pratik Mishra**.

---

## Features

- **Dual-layer detection** — identifies both the orange/red aura and the bright thermal core of a flame, eliminating false positives from posters or clothing.
- **Persistence guard** — fire must be visible for 1.5 seconds continuously before the alarm triggers.
- **Auto-generated alarm** — synthesises a fallback siren if `alarm.mp3` is missing.
- **Clean HUD** — minimal status overlay; no cluttered UI.

---

## Requirements

Python 3.8+ and:

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python3 main.py
```

Press **`q`** to quit.

---

## Tuning (in `main.py`)

| Constant | Default | Purpose |
|---|---|---|
| `MIN_CONTOUR_AREA` | `1500` | Minimum fire blob size (px²). Increase to reduce false positives. |
| `MIN_CORE_RATIO` | `0.07` | Fraction of blob that must be bright core. |
| `PERSISTENCE_THRESHOLD` | `1.5` | Seconds before alarm triggers. |
| `HSV_AURA_LOWER/UPPER` | Orange/Red | Outer flame colour range. |
| `HSV_CORE_LOWER/UPPER` | Yellow/White | Inner hotspot colour range. |

---

## License

MIT License — © 2026 Pratik Mishra. See [LICENSE](LICENSE).
