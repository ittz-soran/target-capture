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

import cv2
import numpy as np

from .arabic import AR, normalize_chars, to_latin_key
from .colors import dominant_color
from .gpu import half

log = logging.getLogger(__name__)

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

_FOLD = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "B": "8",
                       "S": "5", "Z": "2", "G": "6"})


def normalize(text: str) -> str:
    """Canonical plate text: Arabic-Indic/Persian digits -> 0-9, Arabic letter variants unified,
    Latin upper-case, single spaces between groups."""
    text = re.sub(f"[^0-9A-Z{AR} ]", " ", normalize_chars(text))
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
                 car_models=True, memory=None, track_offset=0, skip_empty=False):
        from fast_alpr import ALPR
        from ultralytics import YOLO

        from .gpu import onnx_providers

        self.alpr = ALPR(detector_model=detector_model, ocr_model=ocr_model,
                         detector_conf_thresh=min_det_conf, detector_providers=onnx_providers(),
                         ocr_providers=onnx_providers())
        self.vehicles = YOLO(vehicle_model)
        self.track_offset = track_offset
        self.skip_empty = skip_empty
        self._tracker = _BoxTracker(first_id=track_offset)
        self._tracks: dict[int, dict] = {}   # vehicle track -> make/model found so far, last row saved
        self.vehicle_rows: list[dict] = []   # every vehicle seen, ~1 row per second per vehicle
        self.min_ocr_conf = min_ocr_conf
        self.min_chars = min_chars
        self.vehicle_crops = vehicle_crops
        self._t, self._latin_seen = 0.0, []
        self.memory = memory  # teach.Memory: corrections typed in the Teach tab
        self.min_unread_det = 0.55  # unreadable plates are kept only when the detector is this sure
        # Make / model ("Toyota Corolla") of the vehicle carrying each plate
        self.models = None
        if car_models:
            from .carmodel import CarModelClassifier

            self.models = CarModelClassifier(size=car_models if isinstance(car_models, str) else "auto")
        # script="arabic": read plates with the Arabic + English OCR (Iraq, Kurdistan Region, ...)
        # script="auto": read with the Latin OCR, but check each plate for Arabic script first;
        #   Arabic plates are read with the Arabic OCR, or kept as unread captures if that fails.
        self.script = script
        self.arabic = None
        if script in ("arabic", "auto"):
            from .arabic import ArabicPlateOCR

            self.arabic = ArabicPlateOCR()
            self.min_arabic_conf = min(min_ocr_conf, 0.3)  # EasyOCR scores run lower than fast-alpr's
        # Second reader trained on Iraqi plates: reads two-line plates the general reader gets wrong
        self.iraq = None
        if script == "auto":
            try:
                from .plateocr import PlateOCR

                self.iraq = PlateOCR(providers=onnx_providers())
            except Exception as e:  # model file missing or onnxruntime problem: general reader only
                log.warning("Iraqi plate reader not loaded: %s", e)

    def _vehicles(self, img):
        """Vehicles in the frame with a track id each: [(type, [x1, y1, x2, y2], track_id)]."""
        res = self.vehicles.predict(img, classes=list(VEHICLE_CLASSES), conf=0.3, verbose=False, **half())[0]
        dets = [(VEHICLE_CLASSES[int(cls)], [int(v) for v in box])
                for box, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy())]
        # drop a box lying mostly inside a bigger one (a second detection of part of the same vehicle)
        dets.sort(key=lambda d: -(d[1][2] - d[1][0]) * (d[1][3] - d[1][1]))
        kept = []
        for vtype, b in dets:
            area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
            if all(max(0, min(b[2], k[2]) - max(b[0], k[0])) * max(0, min(b[3], k[3]) - max(b[1], k[1])) / area < 0.8
                   for _, k in kept):
                kept.append((vtype, b))
        return self._tracker.update(kept, self._t)

    def _vehicle_info(self, img, vtype, box, tid) -> dict:
        """Colour of this sighting; make/model a few times per vehicle (first sighting, then when it
        is clearly closer), not on every frame: the model is the slowest step of plate indexing.
        The looks are added up, bigger views counting more, so one odd angle does not decide."""
        x1, y1, x2, y2 = box
        body = img[y1 + (y2 - y1) // 6: y1 + (y2 - y1) // 2, x1 + (x2 - x1) // 6: x2 - (x2 - x1) // 6]
        info = {"vehicle_type": vtype, "vehicle_color": dominant_color(body), "vehicle_model": None,
                "model_conf": None}
        if self.models is None or vtype not in ("car", "truck"):
            return info
        area = (x2 - x1) * (y2 - y1)
        new = {"area": 0, "model": None, "conf": None, "runs": 0, "saved": -1e9, "p": None}
        tr = self._tracks.setdefault(tid, new) if tid is not None else new
        if tr["runs"] == 0 or (area > 1.3 * tr["area"] and tr["runs"] < 4):
            # a little margin around the box: bumpers and mirrors help tell models apart
            mx, my = (x2 - x1) // 20, (y2 - y1) // 20
            p = self.models.probs(img[max(0, y1 - my):y2 + my, max(0, x1 - mx):x2 + mx])
            tr.update(area=area, runs=tr["runs"] + 1)
            if p is not None:
                w = float(np.sqrt(area))
                tr["p"] = p * w if tr["p"] is None else tr["p"] + p * w
                name, conf = self.models.name_of(tr["p"] / tr["p"].sum())
                tr.update(model=name, conf=conf if name else None)
        info.update(vehicle_model=tr["model"], model_conf=tr["conf"])
        return info

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
            if self.iraq is not None:
                text, conf = self._iraq_pick(img, b, text, conf)
            if len(text) < self.min_chars or conf < self.min_ocr_conf:
                # Detected but not readable (e.g. a non-Latin plate, blur, too small): keep the
                # capture so it can be checked by eye.
                if r.detection.confidence >= self.min_unread_det:
                    reads.append({"text": "", "conf": float(r.detection.confidence), "region": None,
                                  "ocr_guess": text or None, **box})
                continue
            text, code_region = strip_codes(text)
            region = code_region
            if self.script == "auto":
                seen = self._known_latin(b)
                if seen is False:
                    arabic = self._arabic_check(img, b)
                    if isinstance(arabic, str):  # English plate; arabic = the words around the number
                        region = arabic or region
                        self._latin_seen.append((self._t, (b.x1 + b.x2) / 2, region))
                    elif (arabic.pop("eastern") or _has_arabic_letters(arabic["text"])
                          or (arabic["text"] and not _latin_wins(text, conf, arabic["text"]))):
                        # an Arabic plate: Arabic read, or an unread capture
                        reads.append({**arabic, **box} if arabic["text"] else
                                     {**arabic, "conf": float(r.detection.confidence), **box})
                        continue
                    else:
                        # English number with Arabic words (e.g. "العراق", "السليمانية") around it:
                        # keep the English read, which is more accurate, and the words as region
                        region = arabic.get("region") or region
                        self._latin_seen.append((self._t, (b.x1 + b.x2) / 2, region))
                else:
                    region = seen or region
            reads.append({"text": text, "conf": conf, "region": region, **box})
        return reads

    def _iraq_pick(self, img, b, text: str, conf: float, sure: float = 0.8) -> tuple[str, float]:
        """General read vs the Iraqi reader. The Iraqi one wins on a sure two-line plate the general
        reader garbled ("21K519K" for 21 K 45194), or when the general read is too weak to use.
        Single-line reads stay with the general reader, which is better there and on foreign plates."""
        from .plateocr import plausible

        iq, iq_conf, layout = self.iraq.read(self._plate_crop(img, b))
        if not iq or iq_conf < sure or not plausible(iq) or len(iq) < self.min_chars:
            return text, conf
        general = strip_codes(text)[0].replace(" ", "")
        if (conf < self.min_ocr_conf or len(text) < self.min_chars
                or (layout == "two-line" and not plausible(general))):
            return iq, iq_conf
        return text, conf

    def _plate_crop(self, img, b):
        x1, y1, x2, y2 = max(b.x1, 0), max(b.y1, 0), min(b.x2, img.shape[1]), min(b.y2, img.shape[0])
        # Small margin: detectors often clip the governorate strip on Iraqi plates.
        mx, my = (x2 - x1) // 20, (y2 - y1) // 8
        return img[max(0, y1 - my):min(img.shape[0], y2 + my), max(0, x1 - mx):min(img.shape[1], x2 + mx)]

    def _known_latin(self, b):
        """The same plate was checked a moment ago and was English: skip the slower Arabic check.
        Saves one Arabic OCR per frame; each car pass is checked about once. Returns False if not
        seen, else the region words found then ('' if none)."""
        cx, w = (b.x1 + b.x2) / 2, max(1, b.x2 - b.x1)
        for t, x, region in reversed(self._latin_seen):
            if self._t - t <= 1.5 and abs(cx - x) <= 3 * w:
                return region or ""
        return False

    def _arabic_check(self, img, b):
        """For an English plate: the region words around the number ('' if none). Otherwise the
        Arabic read (text '' if it could not be read)."""
        from .arabic import has_arabic

        text, conf, details, raw = self.arabic.read(self._plate_crop(img, b))
        region = ", ".join(str(v) for k, v in details.items() if k != "other_text") or None
        if not has_arabic(raw):
            return region or ""
        # number written in Arabic-Indic digits: an Arabic plate even without Arabic letters
        # (3 or more: a lone one is often a misread letter, "E" read as "٤")
        eastern = sum("\u0660" <= c <= "\u0669" or "\u06F0" <= c <= "\u06F9" for c in raw) >= 3
        if sum(c.isdigit() for c in text) >= 3 and conf >= self.min_arabic_conf:
            return {"text": text, "conf": conf, "region": region, "eastern": eastern}
        return {"text": "", "region": region, "ocr_guess": raw or None, "eastern": eastern}

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
        self._latin_seen = [e for e in self._latin_seen if frame.t - e[0] <= 1.5]
        vehicles = self._vehicles(img)
        infos = {}
        for vtype, box, tid in vehicles:
            info = infos[tuple(box)] = self._vehicle_info(img, vtype, box, tid)
            if tid is None:
                continue
            tr = self._tracks.setdefault(tid, {"area": 0, "model": None, "conf": None, "runs": 0, "saved": -1e9})
            if frame.t - tr["saved"] >= 1.0:  # about one row per second per vehicle is enough
                tr["saved"] = frame.t
                self.vehicle_rows.append({"frame": frame.index, "t": frame.t, "track_id": tid,
                                          "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3], **info})
        # Optional: no vehicle in view, skip the plate reader (faster on long recordings of mostly
        # empty road; off by default because the vehicle detector misses some views that the plate
        # detector handles, e.g. steep top-down cameras). Single photos are always read.
        if not vehicles and frame.index > 0 and self.skip_empty:
            return []
        reads = self._read(img)
        if self.vehicle_crops:
            # Two-stage: also read plates inside each vehicle crop. Helps on wide,
            # high-resolution views where plates are tiny relative to the frame.
            for _, (x1, y1, x2, y2), _ in vehicles:
                for r in self._read(img[y1:y2, x1:x2], x1, y1):
                    if not any(_iou(r, o) > 0.3 for o in reads):
                        reads.append(r)
        if not reads:
            return []
        if self.memory:
            for r in reads:
                self.memory.apply(r, img)
        for r in reads:
            r.update(frame=frame.index, t=frame.t, vehicle_type=None, vehicle_color=None,
                     vx1=None, vy1=None, vx2=None, vy2=None,
                     sharp=sharpness(img[max(0, r["y1"]):r["y2"], max(0, r["x1"]):r["x2"]]))
            cx, cy = (r["x1"] + r["x2"]) / 2, (r["y1"] + r["y2"]) / 2
            best = None
            for vtype, (x1, y1, x2, y2), _ in vehicles:
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    area = (x2 - x1) * (y2 - y1)
                    if best is None or area < best[0]:
                        best = (area, (x1, y1, x2, y2))
            if best:
                _, (x1, y1, x2, y2) = best
                r.update(vx1=x1, vy1=y1, vx2=x2, vy2=y2, **infos[(x1, y1, x2, y2)])
        return reads

    def start(self, memory=None, track_offset: int = 0, skip_empty: bool = False):
        """Get ready for a new video, keeping the loaded models (loading them takes seconds)."""
        self.memory = memory
        self.skip_empty = skip_empty
        self.track_offset = track_offset
        self._tracker = _BoxTracker(first_id=track_offset)
        self._tracks, self.vehicle_rows = {}, []
        self._t, self._latin_seen = 0.0, []

    def drain_vehicles(self) -> list[dict]:
        rows, self.vehicle_rows = self.vehicle_rows, []
        return rows


class _BoxTracker:
    """Gives each vehicle a track id across sampled frames. Matches a box to the nearest box of
    the same type seen in the last 2 s (centre within about one vehicle width, similar size);
    simple, and unlike ByteTrack it never drops a detection when frames are far apart."""

    def __init__(self, first_id: int = 0, max_gap: float = 2.0):
        self.next_id, self.max_gap = first_id, max_gap
        self.live: dict[int, tuple] = {}  # id -> (type, box, t)

    def update(self, dets: list[tuple], t: float) -> list[tuple]:
        self.live = {i: v for i, v in self.live.items() if t - v[2] <= self.max_gap}
        pairs = []
        for d, (vtype, b) in enumerate(dets):
            w, h = b[2] - b[0], b[3] - b[1]
            for i, (ltype, lb, lt) in self.live.items():
                lw, lh = lb[2] - lb[0], lb[3] - lb[1]
                dist = np.hypot((b[0] + b[2] - lb[0] - lb[2]) / 2, (b[1] + b[3] - lb[1] - lb[3]) / 2)
                size = max(w * h, lw * lh) / max(1, min(w * h, lw * lh))
                if ltype == vtype and dist <= 1.2 * max(w, lw) and size <= 3:
                    pairs.append((dist / max(w, lw), d, i))
        used_d, used_i, ids = set(), set(), {}
        for _, d, i in sorted(pairs):
            if d not in used_d and i not in used_i:
                used_d.add(d), used_i.add(i)
                ids[d] = i
        out = []
        for d, (vtype, b) in enumerate(dets):
            if d not in ids:
                ids[d] = self.next_id
                self.next_id += 1
            self.live[ids[d]] = (vtype, b, t)
            out.append((vtype, b, ids[d]))
        return out


def _iou(a, b):
    ix = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    union = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# Grouping reads into sightings
# ---------------------------------------------------------------------------

def _has_arabic_letters(text: str) -> bool:
    return bool(re.search(f"[{AR}]", text or ""))


def _latin_wins(latin: str, latin_conf: float, arabic: str) -> bool:
    """English number with Arabic words around it: trust the English read when it is sure and
    its digits agree with the Arabic reader's (it also keeps Latin letters such as "E")."""
    d1, d2 = (re.sub(r"\D", "", x) for x in (latin, arabic))
    return latin_conf >= 0.9 and levenshtein(d1, d2) <= 1


def strip_codes(text: str) -> tuple[str, str | None]:
    """Remove a governorate code the English reader glued to the number ("SUL21E41707" ->
    "21E41707", "Sulaymaniyah"). Only when enough of the number is left."""
    from .arabic import PLATE_CODES

    t = text.replace(" ", "")
    for code, name in PLATE_CODES.items():
        for rest in ((t[3:] if t.startswith(code) else None), (t[:-3] if t.endswith(code) else None)):
            if rest and sum(c.isdigit() for c in rest) >= 4:
                return rest, name
    return text, None


def sharpness(crop) -> float | None:
    """How sharp a plate picture is (motion blur and focus): variance of the Laplacian of the
    plate at a fixed size, so big and small plates compare fairly."""
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 6:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    g = cv2.resize(g, (160, max(8, round(160 * g.shape[0] / g.shape[1]))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _quality(r) -> float:
    """How good a capture is: reader confidence, bigger plates (closer to the camera) and
    sharper pictures (no motion blur) are better. Used to pick the picture shown for a car."""
    if r.get("x1") is None:
        return float(r["conf"] or 0)
    w = max(1, r["x2"] - r["x1"])
    q = float(r["conf"] or 0) * min(1.0, w / 140) ** 0.5
    sharp = r["sharp"] if "sharp" in r.keys() else None  # dict or database row
    if sharp is not None:
        q *= min(1.0, sharp / 400) ** 0.3
    return q


def _near(a, b, secs: float = 1.5) -> bool:
    """Same plate moving through the picture: close in time and position."""
    if a.get("x1") is None or b.get("x1") is None:
        return False
    w = max(1, a["x2"] - a["x1"], b["x2"] - b["x1"])
    ca, cb = ((a["x1"] + a["x2"]) / 2, (a["y1"] + a["y2"]) / 2), ((b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2)
    return abs(b["t"] - a["t"]) <= secs and abs(ca[0] - cb[0]) <= 3 * w and abs(ca[1] - cb[1]) <= 3 * w


def _same_text(a: str, b: str, max_dist: int, near: bool) -> bool:
    if levenshtein(a, b) <= max_dist + (1 if near else 0):
        return True
    # a partial read ("5AU5") of the full plate ("5AU5341")
    short, long_ = sorted((a, b), key=len)
    return short in long_ and len(short) >= (3 if near else 5)


def group_sightings(reads, gap: float = 4.0, max_dist: int = 1) -> list[dict]:
    """Merge reads of the same car into one sighting.

    A read joins a sighting when it is in the same media, within `gap` seconds of the
    sighting's last read, and reads the same (within `max_dist` edits, or as a partial read
    of it). A plate that is still moving through the picture also merges with one more
    difference allowed, so OCR slips do not split one car into several rows.
    """
    open_: dict[int, list[list]] = defaultdict(list)
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
            near = any(_near(x, r) for x in g[-3:])
            if any(_same_text(key, fold(x["text"]), max_dist, near) for x in g[-3:]):
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
        votes[r["text"]] += _quality(r)
    # partial reads support the full text they are part of
    for t in list(votes):
        for r in g:
            if r["text"] != t and fold(r["text"]) in fold(t):
                votes[t] += 0.5 * _quality(r)
    text = max(votes, key=lambda t: (round(votes[t], 6), len(t)))
    same = [r for r in g if r["text"] == text]
    best = max(same, key=_quality)
    # second capture: the best one at least half a second after the best (else before it)
    later = [r for r in same if r["t"] >= best["t"] + 0.5] or [r for r in same if r["t"] <= best["t"] - 0.5]
    second = max(later, key=_quality) if later else None
    vtypes = Counter(r["vehicle_type"] for r in g if r["vehicle_type"])
    vcols = Counter(r["vehicle_color"] for r in g if r["vehicle_color"])
    regions = Counter(r["region"] for r in g if r.get("region"))
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
        "region": regions.most_common(1)[0][0] if regions else None,
        "vehicle_type": vtypes.most_common(1)[0][0] if vtypes else None,
        "vehicle_color": vcols.most_common(1)[0][0] if vcols else None,
        "vehicle_model": model,
        "model_conf": round(model_score / n_model, 2) if model else None,
        "model_alternatives": [m for m, _ in vmodels.most_common(3)[1:]],
        "best": best,
        "second": second,
        "ids": [r["id"] for r in g if "id" in r],
        "source": "taught" if any(r.get("source") == "taught" for r in g)
                  else "learned" if any(r.get("source") == "learned" for r in g) else None,
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
                    "ocr_guess": best.get("ocr_guess"), "ids": [r["id"] for r in g if "id" in r]})
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
    if d > max_dist and 3 <= len(q) < len(p):
        if q in p:  # typed part of the plate exactly ("21L1" in "21L10415")
            return True, 0
        if len(q) < 4:
            return False, d
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
