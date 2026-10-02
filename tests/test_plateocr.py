"""Fast checks for the synthetic plate generator and the Iraqi plate reader: python -m pytest tests"""
import os
import re

import numpy as np
import pytest

from targetcapture import plateocr, synth

PATTERNS = {"A": r"\d{2}[A-Z]\d{5}", "B": r"\d{2}[A-Z]\d{5}", "C": r"\d{9}", "D": r"\d{4,6}"}


@pytest.mark.parametrize("style", synth.STYLES)
def test_render_label_format(style):
    rng = np.random.default_rng(1)
    for _ in range(3):
        img, label = synth.render_plate(rng, style)
        assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
        assert re.fullmatch(PATTERNS[style], label), label
        crop = synth.augment(img, rng)
        assert crop.ndim == 3 and min(crop.shape[:2]) >= 6


def test_prepare_shapes():
    img, _ = synth.render_plate(np.random.default_rng(2), "B")
    cands = plateocr.prepare(img)
    assert len(cands) == len(plateocr.LAYOUTS)
    assert all(c.shape == (1, plateocr.H, plateocr.W) and c.dtype == np.float32 for c in cands)


def test_normalize_label():
    assert plateocr.normalize_label("21 k 45194") == "21K45194"
    assert plateocr.normalize_label("٣٣٩٦٩") == "33969"
    assert plateocr.normalize_label("21 ب 34567") is None


def test_ctc_decode():
    probs = np.full((6, len(plateocr.CHARSET) + 1), 0.01, np.float32)
    for t, k in enumerate([2, 2, 0, 2, 0, 0]):  # "1", "1" after a blank
        probs[t, k] = 0.9
    assert plateocr.ctc_decode(probs)[0] == "11"


@pytest.mark.skipif(not os.path.exists(plateocr.DEFAULT_MODEL), reason="model not trained")
def test_read_types():
    ocr = plateocr.PlateOCR()
    rng = np.random.default_rng(3)
    img, _ = synth.render_plate(rng, "A")
    text, conf, layout = ocr.read(img)
    assert isinstance(text, str) and isinstance(conf, float) and 0.0 <= conf <= 1.0
    assert layout in plateocr.LAYOUTS
    assert ocr.read(np.zeros((2, 2, 3), np.uint8)) == ("", 0.0, "one-line")
