"""Iraqi / Kurdistan Region plate reader: a small CRNN (CTC) trained on synthetic plates.

Two-line plates cannot be read left to right, so every crop is offered to the model in two forms:

* one-line: the whole crop, grey, 32x256;
* two-line: the crop deskewed, trimmed to the text and cut at the gap between the text rows;
  the top and bottom halves (32x128 each) placed side by side.

The model was trained to read the right form and return nothing for the wrong one, so `read`
runs both and keeps the more confident non-empty text. Old plates (Arabic-Indic number over
Arabic governorate words) give only the number. Train with `tools/train_plate_ocr.py`.
"""
from __future__ import annotations

import os
import re

import cv2
import numpy as np

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
H, W = 32, 256
LAYOUTS = ("one-line", "two-line")
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "models", "iraq_plate_ocr.onnx")

_AR_DIGITS = str.maketrans({**{chr(0x0660 + i): str(i) for i in range(10)},
                            **{chr(0x06F0 + i): str(i) for i in range(10)}})
_PLAUSIBLE = [re.compile(p) for p in (r"^\d{2}[A-Z]\d{3,6}$", r"^\d{9}$", r"^\d{3,7}$")]


def normalize_label(text: str) -> str | None:
    """Canonical label (charset only, no spaces), or None when it holds Arabic letters."""
    text = text.translate(_AR_DIGITS).upper()
    if re.search("[؀-ۿݐ-ݿ]", text):
        return None
    return "".join(c for c in text if c in CHARSET)


def _grey(crop: np.ndarray) -> np.ndarray:
    """Grey crop no larger than 256x192 (the model sees 32 rows per text line)."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if g.shape[1] > 256 or g.shape[0] > 192:
        s = min(256 / g.shape[1], 192 / g.shape[0])
        g = cv2.resize(g, (max(8, int(g.shape[1] * s)), max(8, int(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return g


def _norm(g: np.ndarray) -> np.ndarray:
    g = g.astype(np.float32)
    return (g - g.mean()) / (g.std() + 8.0)


def _row_profile(g: np.ndarray) -> np.ndarray:
    return np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)).sum(1)


def deskew_angle(g: np.ndarray) -> float:
    """Rotation (degrees) that makes the text rows horizontal: maximises the sharpness of
    the horizontal projection of vertical edges."""
    h, w = g.shape
    s = 96.0 / w
    small = cv2.resize(g, (96, max(8, int(round(h * s)))), interpolation=cv2.INTER_AREA).astype(np.float32)
    edges = np.abs(cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3))
    sh, sw = edges.shape
    c = (sw / 2, sh / 2)

    def score(a: float) -> float:
        r = cv2.warpAffine(edges, cv2.getRotationMatrix2D(c, a, 1.0), (sw, sh))
        p = r.sum(1)
        return float(np.square(np.diff(p)).sum())

    best = max(np.arange(-20, 21, 4.0), key=score)
    return float(max(np.arange(best - 3, best + 3.1, 1.0), key=score))


def _rotate(g: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.5:
        return g
    h, w = g.shape
    return cv2.warpAffine(g, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0), (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def split_row(g: np.ndarray) -> int:
    """Row between the two text lines: the valley of the edge projection in 30-70% of the height."""
    h = g.shape[0]
    p = _row_profile(g.astype(np.float32))
    k = max(1, h // 12) | 1
    p = np.convolve(p, np.ones(k) / k, mode="same")
    lo, hi = int(h * 0.3), max(int(h * 0.3) + 1, int(h * 0.7))
    return lo + int(np.argmin(p[lo:hi]))


def _span(p: np.ndarray, frac: float) -> tuple[int, int]:
    k = max(1, len(p) // 16) | 1
    p = np.convolve(p, np.ones(k) / k, mode="same")
    idx = np.nonzero(p > frac * p.max())[0] if p.max() > 0 else []
    return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, len(p))


def trim(g: np.ndarray) -> np.ndarray:
    """Cut a loose crop down to the band of strong vertical edges (the text), with a margin."""
    h, w = g.shape
    e = np.abs(cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    y0, y1 = _span(e.sum(1), 0.3)
    x0, x1 = _span(e[y0:y1].sum(0), 0.15)
    my, mx = int((y1 - y0) * 0.1) + 1, int((x1 - x0) * 0.04) + 1
    y0, y1, x0, x1 = max(0, y0 - my), min(h, y1 + my), max(0, x0 - mx), min(w, x1 + mx)
    if y1 - y0 < h * 0.3 or x1 - x0 < w * 0.3:
        return g
    return g[y0:y1, x0:x1]


def one_line(crop: np.ndarray) -> np.ndarray:
    """Whole crop as a (1, 32, 256) float32 model input."""
    return _norm(cv2.resize(_grey(crop), (W, H), interpolation=cv2.INTER_AREA))[None]


def two_line(crop: np.ndarray, deskew: bool = True) -> np.ndarray:
    """Top and bottom text rows side by side as a (1, 32, 256) float32 model input."""
    g = _grey(crop)
    if deskew:
        g = _rotate(g, deskew_angle(g))
    g = trim(g)
    y = split_row(g)
    top = cv2.resize(g[:max(y, 1)], (W // 2, H), interpolation=cv2.INTER_AREA)
    bottom = cv2.resize(g[y:], (W // 2, H), interpolation=cv2.INTER_AREA)
    return _norm(np.hstack([top, bottom]))[None]


def prepare(crop: np.ndarray) -> list[np.ndarray]:
    """Candidate model inputs for a BGR plate crop, in the order of LAYOUTS."""
    g = _grey(crop)
    return [one_line(g), two_line(g)]


def ctc_decode(probs: np.ndarray) -> tuple[str, float]:
    """Greedy CTC decode of (T, 1 + len(CHARSET)) probabilities (blank = 0).
    Confidence: geometric mean of the per-character probabilities (0 for empty text)."""
    best = probs.argmax(1)
    pmax = probs.max(1)
    text, conf, prev = [], [], 0
    for k, p in zip(best, pmax):
        if k != prev and k != 0:
            text.append(CHARSET[k - 1])
            conf.append(p)
        elif k == prev and k != 0:
            conf[-1] = max(conf[-1], p)
        prev = k
    if not text:
        return "", 0.0
    return "".join(text), float(np.exp(np.mean(np.log(np.maximum(conf, 1e-6)))))


def plausible(text: str) -> bool:
    return any(p.match(text) for p in _PLAUSIBLE)


def choose(cands: list[tuple[str, float, str]]) -> tuple[str, float, str]:
    """Most confident non-empty candidate; texts that fit no plate pattern count for less."""
    cands = [c for c in cands if c[0]]
    if not cands:
        return "", 0.0, LAYOUTS[0]
    return max(cands, key=lambda c: c[1] * (1.0 if plausible(c[0]) else 0.7))


class PlateOCR:
    """Reads Iraqi / Kurdistan Region plates (styles: one-line Latin, two-line Latin, black two-line,
    old Arabic-Indic two-tier) from a plate crop."""

    def __init__(self, model_path: str | None = None, providers: list[str] | None = None, threads: int = 2):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model_path or DEFAULT_MODEL, opts,
                                            providers=providers or ["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def candidates(self, crop_bgr: np.ndarray) -> list[tuple[str, float, str]]:
        """(text, confidence, layout) for every input form, in the order of LAYOUTS."""
        x = np.stack(prepare(crop_bgr)).astype(np.float32)
        probs = self.session.run(None, {self.input: x})[0]
        return [(*ctc_decode(p), layout) for p, layout in zip(probs, LAYOUTS)]

    def read(self, crop_bgr: np.ndarray) -> tuple[str, float, str]:
        """Best reading of a BGR plate crop: (text, confidence 0..1, layout "one-line"/"two-line").
        Text is canonical (no spaces, digits 0-9, Latin A-Z); ("", 0.0, "one-line") when nothing
        is read."""
        if crop_bgr is None or crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 4:
            return "", 0.0, LAYOUTS[0]
        return choose(self.candidates(crop_bgr))
