"""Rough colour naming for vehicles and clothing (good enough for "white car", "red top")."""
from __future__ import annotations

import cv2
import numpy as np


def dominant_color(img: np.ndarray) -> str | None:
    if img is None or img.size == 0 or min(img.shape[:2]) < 4:
        return None
    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    h, s, v = hsv[:, 0] * 2, hsv[:, 1] / 255, hsv[:, 2] / 255  # hue in degrees
    names = np.empty(len(h), dtype=object)
    for i in range(len(h)):
        names[i] = _name(h[i], s[i], v[i])
    vals, counts = np.unique(names, return_counts=True)
    return str(vals[counts.argmax()])


def _name(h: float, s: float, v: float) -> str:
    if v < 0.2:
        return "black"
    if s < 0.18:
        return "white" if v > 0.65 else "gray"
    if h < 15 or h >= 345:
        return "red"
    if h < 40:
        return "orange" if v > 0.5 else "brown"
    if h < 70:
        return "yellow"
    if h < 165:
        return "green"
    if h < 255:
        return "blue"
    if h < 290:
        return "purple"
    return "pink"
