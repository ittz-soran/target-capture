"""SQLite index. Indexing a video is slow; searching the index is instant.

One database can hold many videos, so a search covers every indexed camera/file.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3

import cv2
import numpy as np

from .video import SIG_H, SIG_W, read_frame

SCHEMA = """
CREATE TABLE IF NOT EXISTS media (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    kind TEXT, fps REAL, frames INTEGER, width INTEGER, height INTEGER,
    clock_start TEXT,                 -- wall-clock time of frame 0 (DVR time), optional
    plates_next_frame INTEGER DEFAULT 0,   -- resume point for plate indexing
    persons_next_frame INTEGER DEFAULT 0,  -- resume point for person indexing
    plates_done INTEGER DEFAULT 0,
    persons_done INTEGER DEFAULT 0,
    decoder TEXT                      -- 'gpu' or 'cpu': pictures are fetched with the decoder used to index
);
CREATE TABLE IF NOT EXISTS plate_reads (
    id INTEGER PRIMARY KEY,
    media_id INTEGER NOT NULL REFERENCES media(id),
    frame INTEGER, t REAL,
    text TEXT, conf REAL, region TEXT,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
    vehicle_type TEXT, vehicle_color TEXT,
    vehicle_model TEXT, model_conf REAL,   -- e.g. "Toyota Corolla", 0.62
    ocr_guess TEXT,                        -- for plates that could not be read (text = ''): OCR's best guess
    source TEXT,                           -- NULL = read by OCR, 'taught' = typed by the user, 'learned' = from a lesson
    vx1 INTEGER, vy1 INTEGER, vx2 INTEGER, vy2 INTEGER,
    sharp REAL                             -- picture sharpness (no motion blur), to pick the best capture
);
CREATE INDEX IF NOT EXISTS plate_reads_media_t ON plate_reads(media_id, t);
CREATE TABLE IF NOT EXISTS plate_labels (   -- corrections typed in the Teach tab; survive re-indexing
    id INTEGER PRIMARY KEY,
    media_id INTEGER, frame INTEGER, x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
    ocr_text TEXT,          -- what the reader said ('' if it could not read it)
    label TEXT,             -- the correct plate text, '' for "not a plate"
    sig BLOB,               -- small grey picture of the plate (48x144 uint8), for look-alike suggestions
    created TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS frame_sigs (   -- 48x27 grey thumbnail of each frame with a detection
    media_id INTEGER, frame INTEGER, sig BLOB,   -- (finds the exact frame again in videos that seek badly)
    PRIMARY KEY (media_id, frame)
);
CREATE TABLE IF NOT EXISTS vehicle_dets (   -- every vehicle seen (also without a readable plate)
    id INTEGER PRIMARY KEY,
    media_id INTEGER NOT NULL, frame INTEGER, t REAL, track_id INTEGER,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
    vehicle_type TEXT, vehicle_color TEXT, vehicle_model TEXT, model_conf REAL
);
CREATE INDEX IF NOT EXISTS vehicle_dets_media_track ON vehicle_dets(media_id, track_id);
CREATE TABLE IF NOT EXISTS watchlist (      -- wanted plates and faces, checked on every indexed video
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,          -- 'plate' or 'face'
    value TEXT,                  -- plate text (wildcards allowed) or the photo's file name
    note TEXT,
    emb BLOB,                    -- face: 512-d ArcFace embedding
    photo BLOB,                  -- face: small JPEG for the page
    active INTEGER DEFAULT 1,
    created TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY,
    watch_id INTEGER, media_id INTEGER, start REAL, end REAL,
    text TEXT,                   -- what was matched: plate read, or face similarity
    score REAL,
    frame INTEGER, x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
    seen INTEGER DEFAULT 0,
    created TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (watch_id, media_id, start)
);
CREATE TABLE IF NOT EXISTS cameras (        -- live cameras (RTSP), recorded in segments and indexed
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL, url TEXT NOT NULL,
    plates INTEGER DEFAULT 1, persons INTEGER DEFAULT 1, enabled INTEGER DEFAULT 1,
    status TEXT, last_seen TEXT
);
CREATE TABLE IF NOT EXISTS dvrs (           -- DVR / NVR logins saved in the DVR footage tab
    name TEXT PRIMARY KEY, make TEXT, ip TEXT, user TEXT, password TEXT,
    http_port INTEGER DEFAULT 80, rtsp_port INTEGER DEFAULT 554
);
CREATE TABLE IF NOT EXISTS person_dets (
    id INTEGER PRIMARY KEY,
    media_id INTEGER NOT NULL REFERENCES media(id),
    frame INTEGER, t REAL, track_id INTEGER,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
    fx1 INTEGER, fy1 INTEGER, fx2 INTEGER, fy2 INTEGER,  -- face box, NULL if no face
    face_score REAL,
    emb BLOB,                                           -- 512-d float32 ArcFace, L2-normalised
    upper_color TEXT, lower_color TEXT
);
CREATE INDEX IF NOT EXISTS person_dets_media_track ON person_dets(media_id, track_id);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        # timeout: wait for another run (e.g. live cameras) to finish writing instead of failing
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")  # searches can read while indexing writes
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(plate_reads)")}
        for col, typ in (("vehicle_model", "TEXT"), ("model_conf", "REAL"), ("ocr_guess", "TEXT"), ("source", "TEXT"),
                         ("sharp", "REAL")):  # older indexes
            if col not in cols:
                self.db.execute(f"ALTER TABLE plate_reads ADD COLUMN {col} {typ}")
        if "decoder" not in {r["name"] for r in self.db.execute("PRAGMA table_info(media)")}:
            self.db.execute("ALTER TABLE media ADD COLUMN decoder TEXT")
        if "crop" not in {r["name"] for r in self.db.execute("PRAGMA table_info(plate_labels)")}:
            self.db.execute("ALTER TABLE plate_labels ADD COLUMN crop BLOB")  # PNG of the taught plate
        if "face_jpg" not in {r["name"] for r in self.db.execute("PRAGMA table_info(person_dets)")}:
            # small picture of the face, kept while indexing (a person's best faces so far), so
            # the face galleries do not need to find the frame in the video again
            self.db.execute("ALTER TABLE person_dets ADD COLUMN face_jpg BLOB")

    # ---- media -------------------------------------------------------------
    def upsert_media(self, info, clock_start: str | None) -> int:
        path = os.path.abspath(info.path)
        row = self.db.execute("SELECT id FROM media WHERE path=?", (path,)).fetchone()
        if row:
            if clock_start:
                self.db.execute("UPDATE media SET clock_start=? WHERE id=?", (clock_start, row["id"]))
                self.db.commit()
            return row["id"]
        cur = self.db.execute(
            "INSERT INTO media(path, kind, fps, frames, width, height, clock_start) VALUES (?,?,?,?,?,?,?)",
            (path, info.kind, info.fps, info.frames, info.width, info.height, clock_start),
        )
        self.db.commit()
        return cur.lastrowid

    def media(self, media_id: int | None = None):
        if media_id is None:
            return self.db.execute("SELECT * FROM media ORDER BY id").fetchall()
        return self.db.execute("SELECT * FROM media WHERE id=?", (media_id,)).fetchone()

    def remove_media(self, media_id: int) -> str | None:
        """Forget a video: its plates, people and frame pictures. Corrections typed in the Teach
        tab are kept (they still fix the same reading mistakes in other videos)."""
        row = self.media(media_id)
        self.keep_label_crops(media_id)
        for table in ("plate_reads", "person_dets", "frame_sigs", "vehicle_dets", "alerts"):
            self.db.execute(f"DELETE FROM {table} WHERE media_id=?", (media_id,))
        self.db.execute("UPDATE plate_labels SET media_id=NULL WHERE media_id=?", (media_id,))
        self.db.execute("DELETE FROM media WHERE id=?", (media_id,))
        self.db.commit()
        self.purge_orphans()
        self.db.execute("VACUUM")  # give the disk space back
        import shutil

        shutil.rmtree(self.kept_dir(media_id), ignore_errors=True)
        return row["path"] if row else None

    def purge_orphans(self) -> int:
        """Delete rows of videos no longer in the index (e.g. written by a run that was still
        indexing a video while it was removed)."""
        n = 0
        for table in ("plate_reads", "person_dets", "frame_sigs", "vehicle_dets", "alerts"):
            n += self.db.execute(f"DELETE FROM {table} WHERE media_id NOT IN (SELECT id FROM media)").rowcount
        self.db.commit()
        return n

    def set_progress(self, media_id: int, kind: str, next_frame: int, done: bool = False):
        self.db.execute(
            f"UPDATE media SET {kind}_next_frame=?, {kind}_done=? WHERE id=?",
            (next_frame, int(done), media_id),
        )
        self.db.commit()

    def reset(self, media_id: int, kind: str):
        tables = ("plate_reads", "vehicle_dets") if kind == "plates" else ("person_dets",)
        for table in tables:
            self.db.execute(f"DELETE FROM {table} WHERE media_id=?", (media_id,))
        self.db.execute("""DELETE FROM frame_sigs WHERE media_id=? AND frame NOT IN
                           (SELECT frame FROM plate_reads WHERE media_id=? UNION
                            SELECT frame FROM vehicle_dets WHERE media_id=? UNION
                            SELECT frame FROM person_dets WHERE media_id=?)""", (media_id,) * 4)
        self.set_progress(media_id, kind, 0, False)

    # ---- frames ------------------------------------------------------------
    def add_frame_sig(self, media_id: int, frame: int, sig: np.ndarray):
        self.db.execute("INSERT OR REPLACE INTO frame_sigs VALUES (?,?,?)", (media_id, frame, sig.tobytes()))

    def frame_sig(self, row) -> np.ndarray | None:
        r = self.db.execute("SELECT sig FROM frame_sigs WHERE media_id=? AND frame=?",
                            (row["media_id"], row["frame"])).fetchone()
        return None if r is None else np.frombuffer(r["sig"], np.uint8).reshape(SIG_H, SIG_W)

    def kept_dir(self, media_id: int) -> str:
        """Pictures of the frames with plates / faces, kept while indexing (no video to search later)."""
        return os.path.join(os.path.dirname(os.path.abspath(self.path)), "kept_frames", str(media_id))

    def frame(self, row) -> np.ndarray | None:
        """The video frame a detection was made in (exact, also for videos that seek badly)."""
        media = self.media(row["media_id"]) if row["media_id"] is not None else None
        if not media:
            return None
        kept = os.path.join(self.kept_dir(media["id"]), f"{row['frame']:08d}.jpg")
        if os.path.exists(kept):
            img = cv2.imdecode(np.fromfile(kept, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return img
        if os.path.isdir(media["path"]):  # a live camera day: the pictures were kept as they came
            f = os.path.join(media["path"], f"{row['frame']:08d}.jpg")
            try:
                return cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
            except (OSError, ValueError):
                return None
        return read_frame(media["path"], row["frame"], self.frame_sig(row), hw=media["decoder"] == "gpu")

    def decoder(self, media_id: int, wanted: str) -> str:
        """The decoder a video is indexed with: the one already used for it (a resumed or
        repeated run must match the saved frame fingerprints), else `wanted`."""
        row = self.media(media_id)
        used = row["decoder"] if row else None
        has_data = row and (row["plates_next_frame"] or row["persons_next_frame"])
        if used and has_data:
            return used
        self.db.execute("UPDATE media SET decoder=? WHERE id=?", (wanted, media_id))
        self.db.commit()
        return wanted

    # ---- plates ------------------------------------------------------------
    def add_plate_reads(self, media_id: int, rows: list[dict]):
        self.db.executemany(
            """INSERT INTO plate_reads(media_id, frame, t, text, conf, region, x1, y1, x2, y2,
               vehicle_type, vehicle_color, vehicle_model, model_conf, ocr_guess, source, vx1, vy1, vx2, vy2, sharp)
               VALUES (:media_id, :frame, :t, :text, :conf, :region, :x1, :y1, :x2, :y2,
               :vehicle_type, :vehicle_color, :vehicle_model, :model_conf, :ocr_guess, :source, :vx1, :vy1, :vx2, :vy2,
               :sharp)""",
            [{"media_id": media_id, "vehicle_model": None, "model_conf": None, "ocr_guess": None, "source": None,
              "sharp": None, **r}
             for r in rows],
        )

    def plate_reads(self, media_id: int | None = None, unread: bool = False):
        """Plates whose characters were read, or with unread=True the plates that were
        detected but could not be read (text = '')."""
        q = "SELECT * FROM plate_reads WHERE " + ("text = ''" if unread else "text != ''")
        args: tuple = ()
        if media_id is not None:
            q += " AND media_id=?"
            args = (media_id,)
        return self.db.execute(q + " ORDER BY media_id, t", args).fetchall()

    def set_plate_text(self, ids: list[int], text: str, source: str):
        """Overwrite the text of these reads, keeping what the OCR said in ocr_guess."""
        if not ids:
            return
        q = ",".join("?" * len(ids))
        self.db.execute(f"UPDATE plate_reads SET ocr_guess = COALESCE(ocr_guess, NULLIF(text, '')) WHERE id IN ({q})", ids)
        self.db.execute(f"UPDATE plate_reads SET text=?, source=?, conf=1.0 WHERE id IN ({q})", [text, source, *ids])
        self.db.commit()

    def delete_plate_reads(self, ids: list[int]):
        if ids:
            self.db.execute(f"DELETE FROM plate_reads WHERE id IN ({','.join('?' * len(ids))})", ids)
            self.db.commit()

    def reads_with_text(self, text: str):
        return self.db.execute("SELECT * FROM plate_reads WHERE text=? AND source IS NULL", (text,)).fetchall()

    def add_label(self, row, ocr_text: str, label: str, sig, crop: bytes | None = None):
        self.db.execute(
            "INSERT INTO plate_labels(media_id, frame, x1, y1, x2, y2, ocr_text, label, sig, crop) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (row["media_id"], row["frame"], row["x1"], row["y1"], row["x2"], row["y2"], ocr_text, label,
             sig, crop))
        self.db.commit()

    def keep_label_crops(self, media_id: int):
        """Save the plate pictures of this video's taught plates in the index (before the video is
        removed), so they can still be exported for training."""
        import cv2

        for lab in self.db.execute("SELECT * FROM plate_labels WHERE media_id=? AND crop IS NULL AND label != ''",
                                   (media_id,)).fetchall():
            try:
                img = self.frame(lab)
            except Exception:
                img = None
            if img is None:
                continue
            c = img[max(0, lab["y1"]):lab["y2"], max(0, lab["x1"]):lab["x2"]]
            ok, png = cv2.imencode(".png", c) if c.size else (False, None)
            if ok:
                self.db.execute("UPDATE plate_labels SET crop=? WHERE id=?", (png.tobytes(), lab["id"]))

    def labels(self):
        return self.db.execute("SELECT * FROM plate_labels ORDER BY id").fetchall()

    # ---- vehicles ----------------------------------------------------------
    def add_vehicle_dets(self, media_id: int, rows: list[dict]):
        self.db.executemany(
            """INSERT INTO vehicle_dets(media_id, frame, t, track_id, x1, y1, x2, y2, vehicle_type,
               vehicle_color, vehicle_model, model_conf) VALUES (:media_id, :frame, :t, :track_id,
               :x1, :y1, :x2, :y2, :vehicle_type, :vehicle_color, :vehicle_model, :model_conf)""",
            [{"media_id": media_id, "vehicle_model": None, "model_conf": None, **r} for r in rows])

    def vehicle_rows(self, media_id: int | None = None):
        q, args = "SELECT * FROM vehicle_dets", ()
        if media_id is not None:
            q, args = q + " WHERE media_id=?", (media_id,)
        return self.db.execute(q + " ORDER BY media_id, track_id, t", args).fetchall()

    def max_vehicle_track(self, media_id: int) -> int:
        row = self.db.execute("SELECT MAX(track_id) m FROM vehicle_dets WHERE media_id=?", (media_id,)).fetchone()
        return row["m"] or 0

    # ---- watch list and alerts ----------------------------------------------
    def add_watch(self, kind: str, value: str, note: str = "", emb=None, photo: bytes | None = None) -> int:
        cur = self.db.execute("INSERT INTO watchlist(kind, value, note, emb, photo) VALUES (?,?,?,?,?)",
                              (kind, value, note, None if emb is None else np.asarray(emb, np.float32).tobytes(),
                               photo))
        self.db.commit()
        return cur.lastrowid

    def watchlist(self, active_only: bool = False):
        q = "SELECT * FROM watchlist" + (" WHERE active=1" if active_only else "") + " ORDER BY id"
        return self.db.execute(q).fetchall()

    def remove_watch(self, watch_id: int):
        self.db.execute("DELETE FROM alerts WHERE watch_id=?", (watch_id,))
        self.db.execute("DELETE FROM watchlist WHERE id=?", (watch_id,))
        self.db.commit()

    def add_alert(self, a: dict) -> bool:
        """True if this is a new alert (the same match is only alerted once)."""
        cur = self.db.execute(
            """INSERT OR IGNORE INTO alerts(watch_id, media_id, start, end, text, score, frame, x1, y1, x2, y2)
               VALUES (:watch_id, :media_id, :start, :end, :text, :score, :frame, :x1, :y1, :x2, :y2)""", a)
        self.db.commit()
        return cur.rowcount > 0

    def alerts(self, unseen_only: bool = False):
        q = """SELECT a.*, w.kind, w.value, w.note FROM alerts a JOIN watchlist w ON w.id = a.watch_id"""
        return self.db.execute(q + (" WHERE a.seen=0" if unseen_only else "") + " ORDER BY a.id DESC").fetchall()

    def mark_alerts_seen(self):
        self.db.execute("UPDATE alerts SET seen=1")
        self.db.commit()

    # ---- cameras -----------------------------------------------------------
    def add_camera(self, name: str, url: str, plates: bool = True, persons: bool = True) -> int:
        cur = self.db.execute("INSERT INTO cameras(name, url, plates, persons) VALUES (?,?,?,?)",
                              (name, url, int(plates), int(persons)))
        self.db.commit()
        return cur.lastrowid

    def save_dvr(self, name, make, ip, user, password, http_port=80, rtsp_port=554):
        self.db.execute("INSERT OR REPLACE INTO dvrs VALUES (?,?,?,?,?,?,?)",
                        (name, make, ip, user, password, int(http_port), int(rtsp_port)))
        self.db.commit()

    def dvrs(self):
        return self.db.execute("SELECT * FROM dvrs ORDER BY name").fetchall()

    def remove_dvr(self, name):
        self.db.execute("DELETE FROM dvrs WHERE name=?", (name,))
        self.db.commit()

    def cameras(self, enabled_only: bool = False):
        q = "SELECT * FROM cameras" + (" WHERE enabled=1" if enabled_only else "") + " ORDER BY id"
        return self.db.execute(q).fetchall()

    def update_camera(self, cam_id: int, **fields):
        sets = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE cameras SET {sets} WHERE id=?", (*fields.values(), cam_id))
        self.db.commit()

    def remove_camera(self, cam_id: int):
        self.db.execute("DELETE FROM cameras WHERE id=?", (cam_id,))
        self.db.commit()

    # ---- persons -----------------------------------------------------------
    def max_track_id(self, media_id: int) -> int:
        row = self.db.execute("SELECT MAX(track_id) m FROM person_dets WHERE media_id=?", (media_id,)).fetchone()
        return row["m"] or 0

    def add_person_dets(self, media_id: int, rows: list[dict]):
        out = []
        for r in rows:
            r = {"media_id": media_id, **r}
            emb = r.get("emb")
            r["emb"] = None if emb is None else np.asarray(emb, np.float32).tobytes()
            for k in ("fx1", "fy1", "fx2", "fy2", "face_score", "face_jpg"):
                r.setdefault(k, None)
            out.append(r)
        self.db.executemany(
            """INSERT INTO person_dets(media_id, frame, t, track_id, x1, y1, x2, y2,
               fx1, fy1, fx2, fy2, face_score, emb, upper_color, lower_color, face_jpg)
               VALUES (:media_id, :frame, :t, :track_id, :x1, :y1, :x2, :y2,
               :fx1, :fy1, :fx2, :fy2, :face_score, :emb, :upper_color, :lower_color, :face_jpg)""",
            out,
        )

    def face_rows(self, min_face_px: int = 0):
        return self.db.execute(
            "SELECT * FROM person_dets WHERE emb IS NOT NULL AND (fx2 - fx1) >= ? ORDER BY media_id, t",
            (min_face_px,),
        ).fetchall()

    def track_rows(self, media_id: int, track_id: int):
        return self.db.execute(
            "SELECT * FROM person_dets WHERE media_id=? AND track_id=? ORDER BY t", (media_id, track_id)
        ).fetchall()

    def commit(self):
        self.db.commit()


def clock_time(media_row, t: float) -> str | None:
    """Convert a video offset into DVR wall-clock time when --clock-start was given."""
    start = media_row["clock_start"] if media_row is not None else None
    if not start:
        return None
    try:
        base = dt.datetime.fromisoformat(start)
    except ValueError:
        return None
    return (base + dt.timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S")


def fmt_offset(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
