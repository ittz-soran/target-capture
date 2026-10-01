"""Build a synthetic 'DVR' clip: a car with a readable plate drives past twice.

    python tests/make_plate_sample.py car.png out.mp4
"""
import sys

import cv2
import numpy as np

car_path, out = sys.argv[1], sys.argv[2]
car = cv2.imread(car_path)
W, H, FPS, SECONDS = 1280, 720, 25, 60
passes = [(5.0, 9.0), (40.0, 44.0)]  # seconds when the car is in view
rng = np.random.default_rng(0)
bg = np.full((H, W, 3), 90, np.uint8)
cv2.rectangle(bg, (0, 380), (W, H), (60, 60, 60), -1)
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
scale = 1.4
c = cv2.resize(car, None, fx=scale, fy=scale)
for i in range(FPS * SECONDS):
    t = i / FPS
    f = bg.copy()
    for a, b in passes:
        if a <= t <= b:
            x = int(-c.shape[1] + (W + c.shape[1]) * (t - a) / (b - a))
            y = 180
            x0, x1 = max(0, x), min(W, x + c.shape[1])
            if x1 > x0:
                f[y:y + c.shape[0], x0:x1] = c[:, x0 - x:x1 - x]
    noise = rng.integers(0, 4, f.shape, dtype=np.uint8)
    vw.write(cv2.add(f, noise))
vw.release()
print("wrote", out)
