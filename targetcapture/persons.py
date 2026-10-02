"""Person indexing and search.

Indexing: YOLO detects people and ByteTrack links them across frames into tracks
(one track ~ one continuous appearance on camera). For each person box, a face is
looked for in the upper part of the box (cropping first finds much smaller faces
than running a face detector on the whole frame) and turned into a 512-d ArcFace
embedding. Clothing colours are stored as extra details.

Search: the query photo's face embedding is compared (cosine similarity) with every
stored face. A track matches when its best face clears the threshold. Matching
tracks that are close in time are merged (trackers sometimes split one walk into
two tracks), and each merged group is one sighting. The sighting spans the whole
track, so it covers the time the person was on camera even when their face was
turned away.
"""
from __future__ import annotations

import logging
from collections import Counter

import cv2
import numpy as np

from .colors import dominant_color
from .gpu import half

log = logging.getLogger(__name__)


class FaceModel:
    def __init__(self, name="buffalo_l", providers=None):
        from insightface.app import FaceAnalysis

        from .gpu import onnx_providers

        providers = providers or onnx_providers()
        self.app = FaceAnalysis(name=name, providers=list(providers),
                                allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=0 if providers[0] == "CUDAExecutionProvider" else -1, det_size=(640, 640))
        self.det = self.app.det_model
        self.rec = self.app.models["recognition"]

    def faces(self, img, input_size=None):
        """Detect faces and embed them. Returns list of (bbox[4], score, emb[512])."""
        from insightface.app.common import Face

        bboxes, kpss = self.det.detect(img, input_size=input_size, max_num=0)
        out = []
        for i in range(bboxes.shape[0]):
            face = Face(bbox=bboxes[i, :4], kps=kpss[i], det_score=float(bboxes[i, 4]))
            emb = self.rec.get(img, face)
            out.append((bboxes[i, :4], float(bboxes[i, 4]), emb / np.linalg.norm(emb)))
        return out

    def query_embedding(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"cannot read {path}")
        faces = self.faces(img)
        if not faces:
            # Small or tight face crops: pad and retry at a smaller detector size.
            pad = cv2.copyMakeBorder(img, img.shape[0] // 2, img.shape[0] // 2, img.shape[1] // 2,
                                     img.shape[1] // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            faces = self.faces(pad, input_size=(320, 320))
        if not faces:
            raise ValueError(f"no face found in {path}")
        largest = max(faces, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))
        return largest[2]


class PersonIndexer:
    def __init__(self, person_model="yolo11n.pt", tracker="bytetrack.yaml", min_person_px=60,
                 face_model="buffalo_l", min_face_score=0.5, track_offset=0):
        from ultralytics import YOLO

        self.person_model = person_model
        self.yolo = YOLO(person_model)
        self.tracker = tracker
        self.face = FaceModel(face_model)
        self.min_person_px = min_person_px
        self.min_face_score = min_face_score
        self.track_offset = track_offset
        self._untracked = -1
        self._face_try: dict[int, tuple] = {}
        self._faces_seen: dict[int, int] = {}

    def start(self, track_offset: int = 0):
        """Get ready for a new video: fresh tracker, same (slow to load) face model."""
        from ultralytics import YOLO

        self.yolo = YOLO(self.person_model)
        self.track_offset = track_offset
        self._untracked = -1
        self._face_try, self._faces_seen = {}, {}

    def _face_due(self, tid: int, t: float, h: int) -> bool:
        """Look for a face on this person now? Every frame for their first 3 faces, then about
        every 0.6 s, or sooner when they come clearly closer: the same face frame after frame
        adds little, and face detection is the slowest step of person indexing."""
        if tid < 0 or self._faces_seen.get(tid, 0) < 3:
            self._face_try[tid] = (t, h)
            return True
        last_t, last_h = self._face_try.get(tid, (-1e9, 0))
        if t - last_t >= 0.6 or h >= 1.25 * last_h:
            self._face_try[tid] = (t, h)
            return True
        return False

    def _face_in_person(self, img, box):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if h < self.min_person_px:
            return None
        # Head region: top ~45% of the box (or the whole box for close-ups), padded sideways.
        hy2 = y1 + int(h * (0.45 if h > 1.6 * w else 1.0))
        rx1, rx2 = max(0, x1 - w // 5), min(img.shape[1], x2 + w // 5)
        ry1, ry2 = max(0, y1 - h // 20), min(img.shape[0], hy2)
        region = img[ry1:ry2, rx1:rx2]
        if region.size == 0:
            return None
        scale = 1.0
        if min(region.shape[:2]) < 160:  # upscale small heads so the detector sees them
            scale = 160 / min(region.shape[:2])
            region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        faces = [f for f in self.face.faces(region, input_size=(320, 320)) if f[1] >= self.min_face_score]
        if not faces:
            return None
        bb, score, emb = max(faces, key=lambda f: f[1])
        fb = [int(bb[0] / scale + rx1), int(bb[1] / scale + ry1), int(bb[2] / scale + rx1), int(bb[3] / scale + ry1)]
        return fb, score, emb

    def process(self, frame) -> list[dict]:
        img = frame.image
        res = self.yolo.track(img, persist=True, classes=[0], conf=0.35, tracker=self.tracker, verbose=False,
                              **half())[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else None
        rows = []
        for k, (x1, y1, x2, y2) in enumerate(boxes):
            if ids is not None:
                tid = int(ids[k]) + self.track_offset
            else:  # tracker not yet confirmed: keep the detection as its own one-off track
                tid = self._untracked
                self._untracked -= 1
            row = {"frame": frame.index, "t": frame.t, "track_id": tid,
                   "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                   **_clothing(img, int(x1), int(y1), int(x2), int(y2)),
                   "emb": None}
            f = self._face_in_person(img, (int(x1), int(y1), int(x2), int(y2))) \
                if self._face_due(tid, frame.t, int(y2 - y1)) else None
            if f is not None:
                self._faces_seen[tid] = self._faces_seen.get(tid, 0) + 1
                fb, score, emb = f
                row.update(fx1=fb[0], fy1=fb[1], fx2=fb[2], fy2=fb[3], face_score=score, emb=emb)
            rows.append(row)
        return rows


def _clothing(img, x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    if h < 1.6 * w:  # close-up / upper body only: no legs in view
        return {"upper_color": dominant_color(img[y1 + int(h * 0.6): y2, x1 + w // 6: x2 - w // 6]),
                "lower_color": None}
    return {"upper_color": dominant_color(img[y1 + h // 5: y1 + h // 2, x1 + w // 6: x2 - w // 6]),
            "lower_color": dominant_color(img[y1 + int(h * 0.55): y1 + int(h * 0.9), x1 + w // 6: x2 - w // 6])}


def search(store, query_embs: list[np.ndarray], threshold: float = 0.40, merge_gap: float = 3.0,
           min_face_px: int = 20) -> tuple[list[dict], dict]:
    rows = store.face_rows(min_face_px)
    stats = {"faces_indexed": len(rows)}
    if not rows:
        return [], stats
    embs = np.stack([np.frombuffer(r["emb"], dtype=np.float32) for r in rows])
    q = np.stack(query_embs).astype(np.float32)
    sims = (embs @ q.T).max(axis=1)  # best match over all query photos

    # Best face per track
    per_track: dict[tuple, dict] = {}
    for r, s in zip(rows, sims):
        key = (r["media_id"], r["track_id"])
        cur = per_track.get(key)
        if cur is None:
            cur = per_track[key] = {"best": r, "sim": float(s), "hits": 0}
        if s > cur["sim"]:
            cur.update(best=r, sim=float(s))
        if s >= threshold:
            cur["hits"] += 1
    stats["near_misses"] = sum(1 for v in per_track.values() if threshold - 0.08 <= v["sim"] < threshold)

    return _sightings(store, {k: v for k, v in per_track.items() if v["sim"] >= threshold},
                      threshold, merge_gap), stats


def _sightings(store, tracks: dict, threshold: float, merge_gap: float) -> list[dict]:
    """tracks: {(media_id, track_id): {"best": row, "sim": float, "hits": int}} -> sightings."""
    matches = []
    for key, v in tracks.items():
        media_id, tid = key[0], key[1]
        span = store.track_rows(media_id, tid) if tid >= 0 else [v["best"]]
        if len(key) == 3 and "first" in v:  # one segment of a track (see identities)
            span = [r for r in span if v["first"] - 1.0 <= r["t"] <= v["last"] + 1.0] or [v["best"]]
        matches.append({"media_id": media_id, "tracks": [tid], "start": span[0]["t"], "end": span[-1]["t"],
                        "sim": v["sim"], "face_hits": v["hits"], "detections": len(span),
                        "best": v["best"], "span": span})
    matches.sort(key=lambda m: (m["media_id"], m["start"]))

    sightings: list[dict] = []
    for m in matches:
        last = sightings[-1] if sightings else None
        if last and last["media_id"] == m["media_id"] and m["start"] - last["end"] <= merge_gap:
            last["tracks"] += m["tracks"]
            last["end"] = max(last["end"], m["end"])
            last["face_hits"] += m["face_hits"]
            last["detections"] += m["detections"]
            last["span"] += m["span"]
            if m["sim"] > last["sim"]:
                last["sim"], last["best"] = m["sim"], m["best"]
        else:
            sightings.append(m)

    for s in sightings:
        b = s["best"]
        up = Counter(r["upper_color"] for r in s["span"] if r["upper_color"])
        lo = Counter(r["lower_color"] for r in s["span"] if r["lower_color"])
        s["upper_color"] = up.most_common(1)[0][0] if up else None
        s["lower_color"] = lo.most_common(1)[0][0] if lo else None
        s["confidence"] = "high" if s["sim"] >= threshold + 0.15 else "medium" if s["sim"] >= threshold + 0.05 else "low"
        media = store.media(s["media_id"])
        cx = (b["x1"] + b["x2"]) / 2 / max(1, media["width"] if media else 1)
        s["position"] = "left" if cx < 0.33 else "right" if cx > 0.66 else "center"
        del s["span"]
    return sightings


def identities(store, threshold: float = 0.45, min_face_px: int = 24, min_faces: int = 1) -> list[dict]:
    """Group every face in the index into numbered people (no query photo needed).

    Each track (one continuous appearance) gets the average of its face embeddings, then
    tracks are clustered greedily: a track joins the person whose average face it is most
    similar to, if that similarity is at least `threshold`; otherwise it starts a new person.
    People are numbered by first appearance. Returns, per person, its tracks with the
    similarity of each track to the person (for `person_sightings`).
    """
    rows = store.face_rows(min_face_px)
    tracks: dict[tuple, dict] = {}
    # Trackers sometimes hand one ID from a person to another who walks past, so faces are
    # first grouped within each track: a face joins the track segment whose faces it matches
    # (similarity >= 0.3), otherwise it starts a new segment.
    segs: dict[tuple, list[tuple]] = {}
    for r in rows:  # rows are in time order
        base = (r["media_id"], r["track_id"])
        emb = np.frombuffer(r["emb"], dtype=np.float32)
        best_key, best_sim = None, 0.3
        for key in segs.get(base, []):
            t = tracks[key]
            sim = float(emb @ (t["sum"] / np.linalg.norm(t["sum"])))
            if sim >= best_sim:
                best_key, best_sim = key, sim
        if best_key is None:
            best_key = base + (len(segs.get(base, [])),)
            segs.setdefault(base, []).append(best_key)
            tracks[best_key] = {"sum": np.zeros_like(emb), "n": 0, "best": r, "q": -1, "first": r["t"], "last": r["t"]}
        t = tracks[best_key]
        quality = (r["fx2"] - r["fx1"]) * (r["face_score"] or 0)
        t["sum"] = t["sum"] + emb
        t["n"] += 1
        t["last"] = r["t"]
        if quality > t["q"]:
            t["best"], t["q"] = r, quality
    people: list[dict] = []
    for key, t in sorted(tracks.items(), key=lambda kv: -kv[1]["n"]):  # well-seen tracks seed the clusters
        e = t["sum"] / np.linalg.norm(t["sum"])
        sims = [float(e @ (p["sum"] / np.linalg.norm(p["sum"]))) for p in people]
        # One-off faces (tiny, side-on, a single frame) embed less reliably: let them join at a lower bar.
        need = threshold if t["n"] >= 3 else threshold - 0.1
        if sims and max(sims) >= need:
            p = people[int(np.argmax(sims))]
        else:
            p = {"sum": np.zeros_like(e), "tracks": {}, "n_faces": 0, "best": None, "q": -1, "first": t["first"]}
            people.append(p)
        p["sum"] = p["sum"] + e * t["n"]
        p["tracks"][key] = t
        p["n_faces"] += t["n"]
        p["first"] = min(p["first"], t["first"])
        if t["q"] > p["q"]:
            p["best"], p["q"] = t["best"], t["q"]
    people = [p for p in people if p["n_faces"] >= min_faces]
    people.sort(key=lambda p: (p["best"]["media_id"], p["first"]))
    out = []
    for i, p in enumerate(people, 1):
        c = p["sum"] / np.linalg.norm(p["sum"])
        tr = {k: {"best": t["best"], "hits": t["n"], "first": t["first"], "last": t["last"],
                  "sim": float(c @ (t["sum"] / np.linalg.norm(t["sum"])))} for k, t in p["tracks"].items()}
        out.append({"id": i, "n_faces": p["n_faces"], "n_tracks": len(tr), "best": p["best"],
                    "max_track_faces": max(t["n"] for t in p["tracks"].values()),
                    "first": p["first"], "tracks": tr, "embedding": c})
    return out


def person_sightings(store, person: dict, threshold: float = 0.45, merge_gap: float = 3.0) -> list[dict]:
    return _sightings(store, person["tracks"], threshold, merge_gap)

