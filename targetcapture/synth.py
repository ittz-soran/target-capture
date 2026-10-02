"""Synthetic Iraqi / Kurdistan Region licence plates for training the plate reader.

Four styles are rendered (labels are canonical: no spaces, digits 0-9, Latin A-Z):

* A  new plate, one line, white, black text "21 A 21654"          -> "21A21654"
* B  new plate, two lines "21 K" over "45194"                    -> "21K45194"
* C  black plate, white text, two lines "21-22" over "19519"     -> "212219519"
* D  old two-tier plate: Arabic-Indic number on top, governorate and
     "العراق" / type word in two boxes below                      -> top digits only, e.g. "33969"

`augment` turns a flat plate into a camera-like detector crop (tilt, blur, low resolution,
JPEG, noise, shadows, dirt, background around the plate).
"""
from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

STYLES = ("A", "B", "C", "D")
TWO_LINE_STYLES = ("B", "C", "D")
LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"  # series letters; I, O, Q left out (read as 1 / 0)
AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
GOVERNORATES_AR = ["بغداد", "اربيل", "سليمانية", "السليمانية", "دهوك", "كركوك", "البصرة", "نينوى",
                   "الانبار", "ديالى", "بابل", "كربلاء", "النجف", "واسط", "ميسان", "ذي قار", "حلبجة"]
SECOND_WORDS_AR = ["العراق", "العراق", "العراق", "خصوصي", "اجرة", "حمل", "حكومي"]

_SYS = "/usr/share/fonts/truetype"
_OWN = os.path.join(os.path.dirname(__file__), "fonts")
LATIN_FONTS = [
    (f"{_OWN}/BarlowCondensed-SemiBold.ttf", None), (f"{_OWN}/SairaCondensed-SemiBold.ttf", None),
    (f"{_OWN}/RobotoCondensed.ttf", b"Bold"), (f"{_OWN}/RobotoCondensed.ttf", b"SemiBold"),
    (f"{_OWN}/RobotoCondensed.ttf", b"Medium"), (f"{_OWN}/RobotoCondensed.ttf", b"Regular"),
    (f"{_OWN}/Oswald.ttf", b"Medium"), (f"{_OWN}/Oswald.ttf", b"Regular"), (f"{_OWN}/Oswald.ttf", b"Light"),
    (f"{_SYS}/dejavu/DejaVuSans-Bold.ttf", None), (f"{_SYS}/dejavu/DejaVuSansMono-Bold.ttf", None),
    (f"{_SYS}/liberation/LiberationSans-Bold.ttf", None), (f"{_SYS}/freefont/FreeSansBold.ttf", None),
]
LATIN_WEIGHTS = [3, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2]
ARABIC_FONTS = [
    (f"{_SYS}/dejavu/DejaVuSans-Bold.ttf", None), (f"{_SYS}/dejavu/DejaVuSans.ttf", None),
    (f"{_OWN}/NotoSansArabic.ttf", b"Bold"), (f"{_OWN}/NotoSansArabic.ttf", b"Medium"),
    (f"{_OWN}/NotoSansArabic.ttf", b"Black"), (f"{_SYS}/freefont/FreeSerifBold.ttf", None),
]

# BGR (background, text) pairs for old plates
_OLD_COLOURS = [((40, 35, 150), (235, 235, 235)), ((235, 235, 235), (20, 20, 20)),
                ((40, 200, 235), (20, 20, 20)), ((150, 70, 20), (235, 235, 235))]
_OLD_COLOUR_W = [0.35, 0.35, 0.15, 0.15]
_CAR_PAINT = [(240, 240, 240), (200, 200, 200), (30, 30, 30), (90, 90, 90), (150, 150, 150),
              (40, 40, 140), (120, 60, 20), (40, 60, 40), (180, 190, 200), (60, 90, 140)]


@lru_cache(maxsize=None)
def _font(path: str, var: bytes | None, size: int = 96) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
    if var:
        f.set_variation_by_name(var)
    return f


def gov_code(rng: np.random.Generator) -> int:
    """Governorate code of a new plate; 21-24 (Kurdistan Region) dominate the samples."""
    if rng.random() < 0.45:
        return 21
    if rng.random() < 0.3:
        return int(rng.choice([22, 23, 24]))
    return int(rng.integers(11, 36))


def _digits(rng: np.random.Generator, n: int) -> str:
    return "".join(str(int(d)) for d in rng.integers(0, 10, n))


def _text_mask(text: str, font: tuple, rtl: bool = False) -> np.ndarray:
    """Tight grey mask (uint8) of `text` rendered large."""
    f = _font(*font)
    kw = {"direction": "rtl", "language": "ar"} if rtl else {}
    x0, y0, x1, y1 = f.getbbox(text, **kw)
    im = Image.new("L", (x1 - x0 + 4, y1 - y0 + 4), 0)
    ImageDraw.Draw(im).text((2 - x0, 2 - y0), text, fill=255, font=f, **kw)
    m = np.asarray(im)
    ys, xs = np.nonzero(m > 20)
    if len(xs) == 0:
        return m
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _spaced_mask(chars: str, font: tuple, gap: float) -> np.ndarray:
    """Characters drawn one by one, left to right, `gap` x height apart. (Shaping "١ ٣" as one
    string runs the bidi algorithm, which puts the digits in right-to-left order.)"""
    masks = [_text_mask(c, font) for c in chars]
    h = max(m.shape[0] for m in masks)
    sp = np.zeros((h, max(1, int(h * gap))), np.uint8)
    parts = []
    for m in masks:
        top = (h - m.shape[0]) // 2  # Arabic-Indic digits differ in height; centre them
        parts += [np.pad(m, ((top, h - m.shape[0] - top), (0, 0))), sp]
    return np.hstack(parts[:-1])


def _bold(m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r = rng.random()
    if r < 0.2:
        return cv2.dilate(m, np.ones((3, 3), np.uint8))
    if r < 0.4:
        return cv2.erode(m, np.ones((3, 3) if r < 0.3 else (2, 2), np.uint8))
    return m


def _paste(canvas: np.ndarray, mask: np.ndarray, box: tuple, colour, align: str = "c",
           max_stretch: float = 1.25, rng: np.random.Generator | None = None) -> tuple[int, int]:
    """Fit `mask` into box (x0, y0, x1, y1): full box height, natural aspect (optionally stretched),
    shrunk horizontally if wider than the box. Returns the x-range used."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    stretch = 1.0 if rng is None else float(rng.uniform(0.85, max_stretch))
    w = int(round(mask.shape[1] * bh / max(1, mask.shape[0]) * stretch))
    w = max(1, min(w, bw))
    m = cv2.resize(mask, (w, bh), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255.0
    ox = x0 if align == "l" else x1 - w if align == "r" else x0 + (bw - w) // 2
    ox = max(0, min(ox, canvas.shape[1] - w))
    oy = max(0, min(y0, canvas.shape[0] - bh))
    region = canvas[oy:oy + bh, ox:ox + w].astype(np.float32)
    m = m[:region.shape[0], :region.shape[1]]
    canvas[oy:oy + bh, ox:ox + w] = (region * (1 - m) + np.array(colour, np.float32) * m).astype(np.uint8)
    return ox, ox + w


def _plate_base(w: int, h: int, bg, rng: np.random.Generator) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bg
    if rng.random() < 0.6:  # gentle gradient / sheen
        g = np.linspace(rng.uniform(-25, 10), rng.uniform(-10, 25), w if rng.random() < 0.5 else h, dtype=np.float32)
        g = g[None, :, None] if g.shape[0] == w else g[:, None, None]
        img = np.clip(img.astype(np.float32) + g, 0, 255).astype(np.uint8)
    return img


def _strip(img: np.ndarray, sw: int, fg, rng: np.random.Generator) -> None:
    """Narrow left strip of the new plates: separator line and tiny vertical country text."""
    h = img.shape[0]
    t = max(1, int(round(h * rng.uniform(0.012, 0.025))))
    cv2.line(img, (sw, int(h * 0.06)), (sw, int(h * 0.94)), fg, t)
    kind = rng.random()
    if kind < 0.45:
        letters = "IRQ" if rng.random() < 0.7 else "KRG"
        ch = h * rng.uniform(0.11, 0.17)
        y = h * rng.uniform(0.12, 0.25)
        for c in letters:
            m = _text_mask(c, ARABIC_FONTS[0])
            _paste(img, m, (sw * 0.2, y, sw * 0.8, y + ch), fg)
            y += ch * rng.uniform(1.25, 1.7)
    elif kind < 0.8:
        word = rng.choice(["العراق", "كوردستان", "IRAQ", "العراق IRQ"])
        m = np.rot90(_text_mask(str(word), ARABIC_FONTS[0], rtl=not str(word).isascii()))
        _paste(img, m, (sw * 0.2, h * 0.12, sw * 0.8, h * 0.88), fg)
    else:
        third = (20, 20, 20) if rng.random() < 0.5 else (40, 150, 40)  # Iraq / Kurdistan flag bands
        bands = [(30, 30, 200), (240, 240, 240), third]
        y0 = int(h * rng.uniform(0.08, 0.2))
        bh = int(h * rng.uniform(0.08, 0.12))
        for c in bands:
            cv2.rectangle(img, (int(sw * 0.2), y0), (int(sw * 0.8), y0 + bh), c, -1)
            y0 += bh
        m = _text_mask("IRQ", ARABIC_FONTS[0])
        _paste(img, np.rot90(m), (sw * 0.25, y0 + h * 0.05, sw * 0.75, h * 0.9), fg)


def _border(img: np.ndarray, colour, rng: np.random.Generator, p: float = 0.7) -> None:
    if rng.random() < p:
        h, w = img.shape[:2]
        t = max(1, int(round(min(h, w) * rng.uniform(0.01, 0.035))))
        inset = int(round(min(h, w) * rng.uniform(0.0, 0.03)))
        cv2.rectangle(img, (inset, inset), (w - 1 - inset, h - 1 - inset), colour, t)


def _latin_font(rng: np.random.Generator) -> tuple:
    p = np.array(LATIN_WEIGHTS, np.float64)
    return LATIN_FONTS[int(rng.choice(len(LATIN_FONTS), p=p / p.sum()))]


def _new_colours(rng: np.random.Generator):
    v = int(rng.integers(195, 256))
    tint = rng.integers(-12, 8, 3)
    bg = tuple(int(np.clip(v + t, 0, 255)) for t in tint)
    d = int(rng.integers(0, 50))
    fg = tuple(int(np.clip(d + t, 0, 255)) for t in rng.integers(-8, 12, 3))
    return bg, fg


def _style_a(rng: np.random.Generator):
    h = 64
    w = int(h * rng.uniform(4.2, 5.0))
    bg, fg = _new_colours(rng)
    img = _plate_base(w, h, bg, rng)
    sw = int(w * rng.uniform(0.07, 0.10))
    _strip(img, sw, fg, rng)
    code, letter, num = f"{gov_code(rng):02d}", LETTERS[int(rng.integers(len(LETTERS)))], _digits(rng, 5)
    font = _latin_font(rng)
    th = h * rng.uniform(0.58, 0.76)
    ty = (h - th) / 2 + h * rng.uniform(-0.05, 0.05)
    left, right = sw + w * rng.uniform(0.02, 0.05), w * rng.uniform(0.95, 0.985)
    span = right - left
    gap = span * rng.uniform(0.03, 0.07)
    parts = [(code, 2.0), (letter, 1.0), (num, 5.0)]
    unit = (span - 2 * gap) / 8.0
    x = left
    for text, n in parts:
        m = _bold(_text_mask(text, font), rng)
        bw = unit * n
        _paste(img, m, (x, ty, x + bw, ty + th), fg, align="c", rng=rng)
        x += bw + gap
    _border(img, fg, rng, 0.5)
    return img, code + letter + num


def _style_b(rng: np.random.Generator):
    h = 120
    w = int(h * rng.uniform(1.45, 2.0))
    bg, fg = _new_colours(rng)
    img = _plate_base(w, h, bg, rng)
    sw = int(w * rng.uniform(0.1, 0.15))
    _strip(img, sw, fg, rng)
    code, letter, num = f"{gov_code(rng):02d}", LETTERS[int(rng.integers(len(LETTERS)))], _digits(rng, 5)
    font = _latin_font(rng)
    th = h * rng.uniform(0.33, 0.40)
    gap = h * rng.uniform(0.02, 0.08)
    ty = (h - 2 * th - gap) / 2
    left, right = sw + w * rng.uniform(0.03, 0.08), w * rng.uniform(0.92, 0.97)
    span = right - left
    m = _bold(_text_mask(code, font), rng)
    _paste(img, m, (left, ty, left + span * 0.42, ty + th), fg, align="l", rng=rng)
    m = _bold(_text_mask(letter, font), rng)
    lx = left + span * rng.uniform(0.55, 0.8)
    _paste(img, m, (lx, ty, right, ty + th), fg, align="c" if rng.random() < 0.5 else "l", rng=rng)
    m = _bold(_text_mask(num, font), rng)
    _paste(img, m, (left, ty + th + gap, right, ty + 2 * th + gap), fg, align="c", rng=rng)
    _border(img, fg, rng, 0.5)
    return img, code + letter + num


def _style_c(rng: np.random.Generator):
    h = 120
    w = int(h * rng.uniform(1.05, 1.7))
    v = int(rng.integers(0, 45))
    bg = (v, v, v)
    d = int(rng.integers(195, 250))
    fg = tuple(int(np.clip(d + t, 0, 255)) for t in rng.integers(-10, 6, 3))
    img = _plate_base(w, h, bg, rng)
    sw = int(w * rng.uniform(0.1, 0.16))
    _strip(img, sw, fg, rng)
    a, b, num = f"{gov_code(rng):02d}", _digits(rng, 2), _digits(rng, 5)
    font = _latin_font(rng)
    th = h * rng.uniform(0.32, 0.40)
    gap = h * rng.uniform(0.03, 0.1)
    ty = (h - 2 * th - gap) / 2
    left, right = sw + w * rng.uniform(0.04, 0.1), w * rng.uniform(0.86, 0.96)
    top = f"{a}-{b}" if rng.random() < 0.85 else f"{a} {b}"
    m = _bold(_text_mask(top, font), rng)
    _paste(img, m, (left, ty, right, ty + th), fg, align="c", rng=rng)
    m = _bold(_text_mask(num, font), rng)
    _paste(img, m, (left, ty + th + gap, right, ty + 2 * th + gap), fg, align="c", rng=rng)
    if rng.random() < 0.4:  # small mark next to the number
        s = int(h * 0.06)
        x, y = int(right + (w - right) * 0.3), int(ty + 2 * th + gap - s * 2)
        cv2.circle(img, (x, y), max(1, s // 2), fg, -1)
    _border(img, fg, rng, 0.6)
    return img, a + b + num


def _style_d(rng: np.random.Generator):
    h = 120
    w = int(h * rng.uniform(1.5, 2.2))
    i = int(rng.choice(len(_OLD_COLOURS), p=_OLD_COLOURS_P))
    bg, fg = _OLD_COLOURS[i]
    bg = tuple(int(np.clip(c + d, 0, 255)) for c, d in zip(bg, rng.integers(-25, 25, 3)))
    fg = tuple(int(np.clip(c + d, 0, 255)) for c, d in zip(fg, rng.integers(-20, 20, 3)))
    img = _plate_base(w, h, bg, rng)
    t = max(2, int(h * rng.uniform(0.015, 0.03)))
    inset = int(h * rng.uniform(0.0, 0.04))
    cv2.rectangle(img, (inset, inset), (w - 1 - inset, h - 1 - inset), fg, t)
    split = int(h * rng.uniform(0.5, 0.58))
    cv2.line(img, (inset, split), (w - 1 - inset, split), fg, t)
    vx = int(w * rng.uniform(0.45, 0.55))
    cv2.line(img, (vx, split), (vx, h - 1 - inset), fg, t)
    n = int(rng.choice([4, 5, 5, 5, 6, 6]))
    num = _digits(rng, n)
    font = ARABIC_FONTS[int(rng.integers(len(ARABIC_FONTS)))]
    ar = "".join(AR_DIGITS[int(c)] for c in num)
    m = _spaced_mask(ar, font, rng.uniform(0.1, 0.5)) if rng.random() < 0.3 else _text_mask(ar, font)
    m = _bold(m, rng)
    th = (split - inset) * rng.uniform(0.6, 0.8)
    ty = inset + (split - inset - th) / 2 + h * rng.uniform(-0.02, 0.02)
    span = w * rng.uniform(0.5, 0.85)
    x0 = (w - span) / 2
    _paste(img, m, (x0, ty, x0 + span, ty + th), fg, align="c", max_stretch=1.6, rng=rng)
    words = [str(rng.choice(GOVERNORATES_AR)), str(rng.choice(SECOND_WORDS_AR))]
    if rng.random() < 0.5:
        words.reverse()
    bh = (h - inset - split) * rng.uniform(0.45, 0.7)
    by = split + (h - inset - split - bh) / 2
    wfont = ARABIC_FONTS[int(rng.integers(len(ARABIC_FONTS)))]
    for (bx0, bx1), word in zip(((inset, vx), (vx, w - inset)), words):
        m = _bold(_text_mask(word, wfont, rtl=True), rng)
        pad = (bx1 - bx0) * rng.uniform(0.08, 0.18)
        _paste(img, m, (bx0 + pad, by, bx1 - pad, by + bh), fg, align="c", rng=rng)
    return img, num


_OLD_COLOURS_P = np.array(_OLD_COLOUR_W) / sum(_OLD_COLOUR_W)
_RENDER = {"A": _style_a, "B": _style_b, "C": _style_c, "D": _style_d}
STYLE_P = {"A": 0.34, "B": 0.22, "C": 0.2, "D": 0.24}


def render_plate(rng: np.random.Generator, style: str | None = None) -> tuple[np.ndarray, str]:
    """Random flat plate (BGR) and its canonical label. `style` is one of A-D or None (random)."""
    img, label, _ = render_plate_style(rng, style)
    return img, label


def render_plate_style(rng: np.random.Generator, style: str | None = None) -> tuple[np.ndarray, str, str]:
    if style is None:
        style = str(rng.choice(list(STYLE_P), p=list(STYLE_P.values())))
    img, label = _RENDER[style](rng)
    return img, label, style


# ---------------------------------------------------------------- camera realism

def _background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    paint = np.array(_CAR_PAINT[int(rng.integers(len(_CAR_PAINT)))], np.float32) + rng.normal(0, 15, 3)
    img = np.empty((h, w, 3), np.float32)
    img[:] = paint
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img *= rng.uniform(0.7, 1.0) + rng.uniform(0.0, 0.4) * yy
    for _ in range(int(rng.integers(0, 4))):  # bumper bands, chrome, edges
        y0 = int(rng.integers(0, h))
        bh = int(rng.integers(2, max(3, h // 3)))
        c = rng.uniform(10, 60) if rng.random() < 0.6 else rng.uniform(180, 255)
        img[y0:y0 + bh] = img[y0:y0 + bh] * 0.3 + c * 0.7
    if rng.random() < 0.5:
        pts = rng.integers(0, max(h, w), (2, 2))
        cv2.line(img, tuple(int(v) for v in pts[0]), tuple(int(v) for v in pts[1]),
                 tuple(float(c) for c in rng.uniform(0, 255, 3)), int(rng.integers(1, 6)))
    img += rng.normal(0, rng.uniform(1, 8), img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _homography(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    yaw, pitch = np.radians(rng.uniform(-15, 15)), np.radians(rng.uniform(-15, 15))
    roll = np.radians(rng.normal(0, 6) if rng.random() < 0.85 else rng.uniform(-18, 18))
    f = max(w, h) * rng.uniform(1.2, 3.0)
    pts = np.c_[src - [w / 2, h / 2], np.zeros(4)]
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    p = pts @ (rz @ rx @ ry).T
    p[:, 2] += f
    dst = p[:, :2] * (f / p[:, 2:3])
    dst[:, 0] *= rng.uniform(0.9, 1.1)
    if rng.random() < 0.3:  # shear: text leaning as seen on yawed plates after deskew
        dst[:, 0] += rng.uniform(-0.3, 0.3) * (dst[:, 1] - dst[:, 1].mean())
    dst -= dst.min(0)
    return src, dst.astype(np.float32)


def _motion_kernel(k: int, rng: np.random.Generator) -> np.ndarray:
    ker = np.zeros((k, k), np.float32)
    ker[k // 2] = 1
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
    ker = cv2.warpAffine(ker, rot, (k, k))
    return ker / max(ker.sum(), 1e-6)


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Camera-like crop of a flat plate: as a plate detector would cut it out of a video frame."""
    h, w = img.shape[:2]
    text_rows = 0.7 if w / h > 3 else 0.37
    target_text = rng.uniform(9, 45) if rng.random() < 0.85 else rng.uniform(45, 90)
    pre = min(1.0, 1.6 * target_text / (text_rows * h))
    if pre < 0.9:  # work at ~2x the final resolution
        img = cv2.resize(img, (max(8, int(w * pre)), max(8, int(h * pre))), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    if rng.random() < 0.3:  # dirt / wear on the plate itself
        img = img.copy()
        for _ in range(int(rng.integers(1, 6))):
            c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            ax = (int(rng.integers(2, max(3, w // 6))), int(rng.integers(2, max(3, h // 4))))
            over = img.copy()
            cv2.ellipse(over, c, ax, float(rng.uniform(0, 180)), 0, 360,
                        tuple(int(v) for v in rng.integers(40, 120, 3)), -1)
            a = rng.uniform(0.1, 0.4)
            img = cv2.addWeighted(over, a, img, 1 - a, 0)
    src, dst = _homography(w, h, rng)
    pw, ph = dst[:, 0].max(), dst[:, 1].max()
    # detector box: bounding box of the tilted plate, loose or tight
    if rng.random() < 0.7:
        m = rng.uniform(-0.03, 0.2, 4)
    else:
        m = rng.uniform(0.0, 0.5, 4)
    pad = np.array([m[0] * pw, m[1] * ph, m[2] * pw, m[3] * ph])
    cw, ch = int(round(pw + pad[0] + pad[2])), int(round(ph + pad[1] + pad[3]))
    cw, ch = max(cw, 8), max(ch, 8)
    dst = dst + np.float32([pad[0], pad[1]])
    H = cv2.getPerspectiveTransform(src, dst)
    bg = _background(ch, cw, rng)
    warped = cv2.warpPerspective(img, H, (cw, ch), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (cw, ch), flags=cv2.INTER_LINEAR)
    if rng.random() < 0.5:  # plate frame / holder edge
        cv2.polylines(bg, [dst.astype(np.int32)], True, tuple(int(v) for v in rng.integers(0, 80, 3)),
                      int(rng.integers(2, 6)))
    a = mask.astype(np.float32)[..., None] / 255.0
    out = (warped * a + bg * (1 - a)).astype(np.float32)

    if rng.random() < 0.4:  # shadow
        poly = rng.integers(0, [cw, ch], (int(rng.integers(3, 6)), 2)).astype(np.int32)
        sm = np.zeros((ch, cw), np.float32)
        cv2.fillPoly(sm, [poly], 1.0)
        sm = cv2.GaussianBlur(sm, (0, 0), max(1.0, cw * 0.02))
        out *= 1 - rng.uniform(0.2, 0.6) * sm[..., None]
    if rng.random() < 0.2:  # glare
        c = (int(rng.integers(0, cw)), int(rng.integers(0, ch)))
        gm = np.zeros((ch, cw), np.float32)
        cv2.circle(gm, c, int(rng.integers(3, max(4, cw // 4))), 1.0, -1)
        gm = cv2.GaussianBlur(gm, (0, 0), max(1.0, cw * 0.05))
        out += rng.uniform(40, 120) * gm[..., None]
    # colour / brightness / contrast
    out = out * rng.uniform(0.6, 1.3) + rng.uniform(-40, 40)
    out = out * (1 + rng.normal(0, 0.06, 3)) + rng.normal(0, 6, 3)
    mean = out.mean()
    out = (out - mean) * rng.uniform(0.5, 1.2) + mean
    out = np.clip(out, 0, 255)
    if rng.random() < 0.3:
        g = rng.uniform(0.6, 1.6)
        out = 255.0 * (out / 255.0) ** g
    out = out.astype(np.uint8)
    if rng.random() < 0.2:
        out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    # resolution: plate text height down to ~10 px, then back up (video frames are upscaled)
    scale = target_text / (text_rows * ph)
    small = (max(8, int(round(cw * scale))), max(6, int(round(ch * scale))))
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.5) / scale ** 0.5)
    if rng.random() < 0.3:
        k = int(rng.integers(3, 9))
        out = cv2.filter2D(out, -1, _motion_kernel(k, rng))
    out = cv2.resize(out, small, interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        n = rng.normal(0, rng.uniform(2, 12), out.shape)
        out = np.clip(out + n, 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        q = int(rng.integers(15, 90))
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if rng.random() < 0.6 and out.shape[0] < 100:
        up = min(rng.uniform(1.5, 6.0), 128.0 / out.shape[0])
        out = cv2.resize(out, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        if rng.random() < 0.5:
            bl = cv2.GaussianBlur(out, (0, 0), rng.uniform(1, 3))
            out = cv2.addWeighted(out, 1.6, bl, -0.6, 0)
    if rng.random() < 0.4:  # soft, smoothed video look
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.02, 0.07) * text_rows * out.shape[0])
    return out
