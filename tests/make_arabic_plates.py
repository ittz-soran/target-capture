"""Render Iraq / Kurdistan style test plates and paste them onto a car photo.

    python tests/make_arabic_plates.py car.png out_dir/
Writes plate_*.png (flat plates) and car_*.png (plate on the car) for each sample.
Needs Pillow with libraqm for Arabic shaping and a font with Arabic glyphs (DejaVu Sans).
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = os.environ.get("PLATE_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
AR = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

# (name, top-left label, top-right label, number, series letter, layout)
SAMPLES = [
    ("erbil_private", "اربيل", "خصوصي", "12345", "", "kurdistan"),
    ("baghdad_new", "IRQ", "بغداد", "34567", "ب", "iraq_new"),
    ("duhok_taxi", "دهوك", "اجرة", "70981", "", "kurdistan"),
]


def text(d, xy, s, size, anchor="mm", fill=(0, 0, 0)):
    d.text(xy, s, font=ImageFont.truetype(FONT, size), fill=fill, anchor=anchor, direction=None)


def render(top_l, top_r, number, letter, layout):
    W, H = 520, 130 if layout == "iraq_new" else 220
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, W - 3, H - 3], outline=(0, 0, 0), width=4)
    if layout == "kurdistan":
        d.rectangle([4, 4, W - 5, 60], fill=(200, 30, 30))
        text(d, (W * 0.25, 32), top_r, 34, fill=(255, 255, 255))
        text(d, (W * 0.75, 32), top_l, 34, fill=(255, 255, 255))
        text(d, (W / 2, 105), number.translate(AR), 66)
        text(d, (W / 2, 180), number, 58)
    else:
        d.rectangle([4, 4, 80, H - 5], fill=(20, 60, 160))
        text(d, (42, H / 2), top_l, 24, fill=(255, 255, 255))
        text(d, (W * 0.58, 26), top_r, 26)
        text(d, (W * 0.58, 85), f"{number}  {letter}  21", 60)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main(car_path, out):
    os.makedirs(out, exist_ok=True)
    car = cv2.imread(car_path)
    for name, tl, tr, num, let, layout in SAMPLES:
        plate = render(tl, tr, num, let, layout)
        cv2.imwrite(os.path.join(out, f"plate_{name}.png"), plate)
        # Paste over the original plate of the test car (centred, 120 px wide), on a 2x upscaled car.
        c = cv2.resize(car, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        pw = 170
        ph = int(plate.shape[0] * pw / plate.shape[1])
        p = cv2.resize(plate, (pw, ph), interpolation=cv2.INTER_AREA)
        cx, cy = 694, 397
        c[cy - ph // 2: cy - ph // 2 + ph, cx - pw // 2: cx - pw // 2 + pw] = p
        cv2.imwrite(os.path.join(out, f"car_{name}.png"), c)
    print("wrote samples to", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
