"""License plate indexing and search.

Indexing: every sampled frame goes through a plate detector + plate OCR (fast-alpr,
ONNX). When a plate is found, a YOLO model labels the vehicle carrying it (car,
truck, bus, motorcycle) and its colour. Every read is stored.

Search: OCR on CCTV is noisy, so one car usually produces several slightly
different reads ("12ABC345", "12A8C345", "I2ABC345"). Searching therefore
  1. compares plates after folding look-alike characters (O/0/D/Q, I/1/L, B/8, S/5, Z/2, G/6),
  2. allows a small edit distance, and supports wildcards ("12?BC*"),
  3. merges reads that are close in time into one sighting, with a consensus text.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict

import numpy as np

from .arabic import normalize_chars, to_latin_key
from .colors import dominant_color

log = logging.getLogger(__name__)

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

_FOLD = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "B": "8",
                       "S": "5", "Z": "2", "G": "6"})


def normalize(text: str) -> str:
    """Canonical plate text: Arabic-Indic/Persian digits -> 0-9, Arabic letter variants unified,
    Latin upper-case, single spaces between groups."""
    text = re.sub(r"[^0-9A-Z\u0600-\u06FF ]", " ", normalize_chars(text))
    return " ".join(text.split())


def fold(text: str) -> str:
    """Matching key: Arabic letters to a Latin key, look-alike characters merged, no spaces."""
    return to_latin_key(normalize(text)).replace(" ", "").translate(_FOLD)


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class PlateIndexer:
    def __init__(self, detector_model="yolo-v9-t-384-license-plate-end2end",
                 ocr_model="cct-s-v2-global-model", vehicle_model="yolo11n.pt",
                 min_det_conf=0.4, min_ocr_conf=0.5, min_chars=4, vehicle_crops=False, script="auto",
                 car_models=True):
        from fast_alpr import ALPR
        from ultralytics import YOLO

        self.alpr = ALPR(detector_model=detector_model, ocr_model=ocr_model,
                         detector_conf_thresh=min_det_conf)
        self.vehicles = YOLO(vehicle_model)
        self.min_ocr_conf = min_ocr_conf
        self.min_chars = min_chars
        self.vehicle_crops = vehicle_crops
        self._t, self._latin_seen = 0.0, []
        self.min_unread_det = 0.55  # unreadable plates are kept only when the detector is this sure
        # Make / model ("Toyota Corolla") of the vehicle carrying each plate
        self.models = None
        if car_models:
            from .carmodel import CarModelClassifier

            self.models = CarModelClassifier()
        # script="arabic": read plates with the Arabic + English OCR (Iraq, Kurdistan Region, ...)
        # script="auto": read with the Latin OCR, but check each plate for Arabic script first;
        #   Arabic plates are read with the Arabic OCR, or kept as unread captures if that fails.
        self.script = script
        self.arabic = None
        if script in ("arabic", "auto"):
            from .arabic import ArabicPlateOCR

            self.arabic = ArabicPlateOCR()
            self.min_arabic_conf = min(min_ocr_conf, 0.3)  # EasyOCR scores run lower than fast-alpr's

    def _vehicles(self, img):
        res = self.vehicles.predict(img, classes=list(VEHICLE_CLASSES), conf=0.3, verbose=False)[0]
        out = []
        for box, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            out.append((VEHICLE_CLASSES[int(cls)], [int(v) for v in box]))
        return out

    def _read(self, img, ox=0, oy=0):
        if self.script == "arabic":
            return self._read_arabic(img, ox, oy)
        reads = []
        for r in self.alpr.predict(img):
            if r.ocr is None:
                continue
            text = normalize(r.ocr.text or "")
            conf = r.ocr.confidence
            conf = float(np.mean(conf)) if isinstance(conf, (list, tuple, np.ndarray)) else float(conf or 0)
            b = r.detection.bounding_box
            box = {"x1": b.x1 + ox, "y1": b.y1 + oy, "x2": b.x2 + ox, "y2": b.y2 + oy}
            if len(text) < self.min_chars or conf < self.min_ocr_conf:
                # Detected but not readable (e.g. a non-Latin plate, blur, too small): keep the
                # capture so it can be checked by eye.
                if r.detection.confidence >= self.min_unread_det:
                    reads.append({"text": "", "conf": float(r.detection.confidence), "region": None,
                                  "ocr_guess": text or None, **box})
                continue
            if self.script == "auto" and not self._known_latin(b):
                arabic = self._arabic_check(img, b)
                if arabic is not None:  # not an English plate: Arabic read, or an unread capture
                    reads.append({**arabic, **box} if arabic["text"] else
                                 {**arabic, "conf": float(r.detection.confidence), **box})
                    continue
            reads.append({"text": text, "conf": conf, "region": getattr(r.ocr, "region", None), **box})
        return reads

    def _plate_crop(self, img, b):
        x1, y1, x2, y2 = max(b.x1, 0), max(b.y1, 0), min(b.x2, img.shape[1]), min(b.y2, img.shape[0])
        # Small margin: detectors often clip the governorate strip on Iraqi plates.
        mx, my = (x2 - x1) // 20, (y2 - y1) // 8
        return img[max(0, y1 - my):min(img.shape[0], y2 + my), max(0, x1 - mx):min(img.shape[1], x2 + mx)]

    def _known_latin(self, b) -> bool:
        """The same plate was checked a moment ago and was English: skip the slower Arabic check.
        Saves one Arabic OCR per frame; each car pass is checked about once."""
        cx, w = (b.x1 + b.x2) / 2, max(1, b.x2 - b.x1)
        return any(self._t - t <= 1.5 and abs(cx - x) <= 3 * w for t, x in self._latin_seen)

    def _arabic_check(self, img, b):
        """None for a Latin plate; otherwise the Arabic read (text '' if it could not be read)."""
        from .arabic import has_arabic

        text, conf, details, raw = self.arabic.read(self._plate_crop(img, b))
        if not has_arabic(raw):
            self._latin_seen.append((self._t, (b.x1 + b.x2) / 2))
            return None
        region = ", ".join(str(v) for k, v in details.items() if k != "other_text") or None
        if sum(c.isdigit() for c in text) >= 3 and conf >= self.min_arabic_conf:
            return {"text": text, "conf": conf, "region": region}
        return {"text": "", "region": region, "ocr_guess": raw or None}

    def _read_arabic(self, img, ox=0, oy=0):
        reads = []
        for det in self.alpr.detector.predict(img):
            b = det.bounding_box
            x1, y1, x2, y2 = max(b.x1, 0), max(b.y1, 0), min(b.x2, img.shape[1]), min(b.y2, img.shape[0])
            text, conf, details, raw = self.arabic.read(self._plate_crop(img, b))
            digits = sum(c.isdigit() for c in text)
            if digits < 3 or len(text.replace(" ", "")) < self.min_chars - 1 or conf < self.min_arabic_conf:
                if det.confidence >= self.min_unread_det:
                    reads.append({"text": "", "conf": float(det.confidence), "region": None,
                                  "ocr_guess": raw or None,
                                  "x1": x1 + ox, "y1": y1 + oy, "x2": x2 + ox, "y2": y2 + oy})
                continue
            region = ", ".join(str(v) for k, v in details.items() if k != "other_text") or None
            reads.append({"text": text, "conf": conf, "region": region,
                          "x1": x1 + ox, "y1": y1 + oy, "x2": x2 + ox, "y2": y2 + oy})
        return reads

    def process(self, frame) -> list[dict]:
        img = frame.image
        self._t = frame.t
        self._latin_seen = [(t, x) for t, x in self._latin_seen if frame.t - t <= 1.5]
        vehicles = None
        reads = self._read(img)
        if self.vehicle_crops:
            # Two-stage: also read plates inside each vehicle crop. Helps on wide,
            # high-resolution views where plates are tiny relative to the frame.
            vehicles = self._vehicles(img)
            for _, (x1, y1, x2, y2) in vehicles:
                for r in self._read(img[y1:y2, x1:x2], x1, y1):
                    if not any(_iou(r, o) > 0.3 for o in reads):
                        reads.append(r)
        if not reads:
            return []
        if vehicles is None:
            vehicles = self._vehicles(img)
        for r in reads:
            r.update(frame=frame.index, t=frame.t, vehicle_type=None, vehicle_color=None,
                     vx1=None, vy1=None, vx2=None, vy2=None)
            cx, cy = (r["x1"] + r["x2"]) / 2, (r["y1"] + r["y2"]) / 2
            best = None
            for vtype, (x1, y1, x2, y2) in vehicles:
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    area = (x2 - x1) * (y2 - y1)
                    if best is None or area < best[0]:
                        best = (area, vtype, (x1, y1, x2, y2))
            if best:
                _, vtype, (x1, y1, x2, y2) = best
                # Colour from the upper body of the vehicle, avoiding road and plate.
                body = img[y1 + (y2 - y1) // 6: y1 + (y2 - y1) // 2, x1 + (x2 - x1) // 6: x2 - (x2 - x1) // 6]
                r.update(vehicle_type=vtype, vehicle_color=dominant_color(body),
                         vx1=x1, vy1=y1, vx2=x2, vy2=y2)
                if self.models is not None and vtype in ("car", "truck"):
                    name, p = self.models.best(img[y1:y2, x1:x2])
                    r.update(vehicle_model=name, model_conf=p if name else None)
        return reads


def _iou(a, b):
    ix = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    union = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# Grouping reads into sightings
# ---------------------------------------------------------------------------

def group_sightings(reads, gap: float = 4.0, max_dist: int = 1) -> list[dict]:
    """Merge reads of the same plate that are close in time into one sighting.

    Two reads join the same sighting when they are in the same media, within `gap`
    seconds of the sighting's last read, and their folded texts are within
    `max_dist` edits.
    """
    open_: dict[int, list[dict]] = defaultdict(list)
    done: list[list] = []
    for r in reads:
        r = dict(r)
        key = fold(r["text"])
        groups = open_[r["media_id"]]
        still_open = []
        for g in groups:
            if r["t"] - g[-1]["t"] > gap:
                done.append(g)
            else:
                still_open.append(g)
        groups[:] = still_open
        target = None
        for g in groups:
            if levenshtein(key, fold(g[-1]["text"])) <= max_dist:
                target = g
                break
        if target is None:
            groups.append([r])
        else:
            target.append(r)
    for groups in open_.values():
        done.extend(groups)
    return sorted((_summarize(g) for g in done), key=lambda s: (s["media_id"], s["start"]))


def _summarize(g: list[dict]) -> dict:
    votes = Counter()
    for r in g:
        votes[r["text"]] += r["conf"]
    text = votes.most_common(1)[0][0]
    best = max(g, key=lambda r: (r["text"] == text, r["conf"]))
    vtypes = Counter(r["vehicle_type"] for r in g if r["vehicle_type"])
    vcols = Counter(r["vehicle_color"] for r in g if r["vehicle_color"])
    vmodels = Counter()
    for r in g:
        if r.get("vehicle_model"):
            vmodels[r["vehicle_model"]] += r["model_conf"] or 0
    model, model_score = vmodels.most_common(1)[0] if vmodels else (None, 0)
    n_model = sum(1 for r in g if r.get("vehicle_model") == model)
    return {
        "media_id": g[0]["media_id"],
        "text": text,
        "variants": sorted({r["text"] for r in g}),
        "start": g[0]["t"], "end": g[-1]["t"],
        "reads": len(g),
        "conf": round(best["conf"], 3),
        "region": best["region"],
        "vehicle_type": vtypes.most_common(1)[0][0] if vtypes else None,
        "vehicle_color": vcols.most_common(1)[0][0] if vcols else None,
        "vehicle_model": model,
        "model_conf": round(model_score / n_model, 2) if model else None,
        "model_alternatives": [m for m, _ in vmodels.most_common(3)[1:]],
        "best": best,
    }


def group_unread(rows, gap: float = 2.0) -> list[dict]:
    """Group detections of plates that could not be read into captures (one per car pass).

    Consecutive detections in the same media join when they are within `gap` seconds and
    the plate has not jumped further than a few plate widths. Each capture keeps its
    clearest detection (largest plate, most confident detector) to show as the picture.
    """
    groups: list[list] = []
    open_: dict[int, list[list]] = defaultdict(list)
    for r in rows:
        r = dict(r)
        cx, w = (r["x1"] + r["x2"]) / 2, max(1, r["x2"] - r["x1"])
        target = None
        for g in open_[r["media_id"]]:
            last = g[-1]
            lcx = (last["x1"] + last["x2"]) / 2
            if r["t"] - last["t"] <= gap and abs(cx - lcx) <= 4 * w:
                target = g
                break
        if target is None:
            target = []
            open_[r["media_id"]].append(target)
            groups.append(target)
        target.append(r)
        open_[r["media_id"]] = [g for g in open_[r["media_id"]] if r["t"] - g[-1]["t"] <= gap]
    out = []
    for g in groups:
        best = max(g, key=lambda r: (r["x2"] - r["x1"]) * (r["y2"] - r["y1"]) * r["conf"])
        models = Counter(r["vehicle_model"] for r in g if r.get("vehicle_model"))
        out.append({"media_id": g[0]["media_id"], "start": g[0]["t"], "end": g[-1]["t"], "detections": len(g),
                    "best": best, "vehicle_type": best["vehicle_type"], "vehicle_color": best["vehicle_color"],
                    "vehicle_model": models.most_common(1)[0][0] if models else None,
                    "ocr_guess": best.get("ocr_guess")})
    return sorted(out, key=lambda c: (c["media_id"], c["start"]))


# ---------------------------------------------------------------------------
# Query matching
# ---------------------------------------------------------------------------

def match_score(query: str, plate: str, max_dist: int) -> tuple[bool, int]:
    """Return (matched, distance). Supports '?' (one char) and '*' (any run) wildcards."""
    if "?" in query or "*" in query:
        pattern = "".join("." if c == "?" else ".*" if c == "*" else re.escape(fold(c))
                          for c in query.upper() if c in "?*" or c.isalnum())
        return bool(re.fullmatch(pattern, fold(plate))), 0
    q, p = fold(query), fold(plate)
    d = levenshtein(q, p)
    # Only part of the plate known (e.g. the number without the governorate code or series
    # letter): compare against every same-length stretch of the plate.
    if d > max_dist and 4 <= len(q) < len(p):
        allowed = max_dist if len(q) >= 5 else 0
        part = min(levenshtein(q, p[i:i + len(q)]) for i in range(len(p) - len(q) + 1))
        if part <= allowed:
            return True, part
    return d <= max_dist, d


def search(store, query: str, max_dist: int = 1, gap: float = 4.0, min_reads: int = 1) -> list[dict]:
    reads = store.plate_reads()
    hits = []
    for s in group_sightings(reads, gap=gap):
        if s["reads"] < min_reads:
            continue
        best_d = None
        for v in s["variants"]:
            ok, d = match_score(query, v, max_dist)
            if ok and (best_d is None or d < best_d):
                best_d = d
        if best_d is not None:
            s["distance"] = best_d
            s["exact"] = normalize(query).replace(" ", "") == s["text"].replace(" ", "")
            s["partial"] = len(fold(query)) < len(fold(s["text"])) and "*" not in query
            hits.append(s)
    return hits
