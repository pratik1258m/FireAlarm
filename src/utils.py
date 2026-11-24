"""
Utility functions for Fire Detection System.
"""
from __future__ import annotations


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamps a numeric value between low and high bounds."""
    return float(min(high, max(low, value)))
