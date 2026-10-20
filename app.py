"""Web UI: upload a video or image, index it, then search plates or people.

    streamlit run app.py
"""
import datetime as dt
import os
import shutil
import subprocess
import sys
import tempfile

import streamlit as st

from targetcapture import persons as P
from targetcapture import settings as C
from targetcapture import plates as L
from targetcapture import teach as T
from targetcapture import vehicles as V
from targetcapture import watch as W
from targetcapture.export import to_excel, to_pdf
from targetcapture.pipeline import index_media
from targetcapture.report import alert_report, person_report, plate_report, thumbnail, vehicle_report
from targetcapture.store import Store, clock_time, fmt_offset

UPLOADS = "uploads"
RESULTS = "results"
THUMBS = os.path.join(RESULTS, "_thumbs")
CFG = C.load()
PAGE = int(CFG["page_size"])
SCRIPTS = ["auto", "latin", "arabic"]
APP_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Target Capture", layout="wide")


def from_this_pc() -> bool:
    ip = st.context.ip_address
    return ip is None or ip.startswith("127.") or ip in ("::1", "localhost")


# Opened from another device (phone, office, Tailscale): ask the page password set in Settings
if CFG["page_password"] and not from_this_pc() and not st.session_state.get("signed_in"):
    st.title("Target Capture")
    with st.form("sign_in"):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            if C.check_password(pw, CFG["page_password"]):
                st.session_state["signed_in"] = True
                st.rerun()
            st.error("Wrong password.")
    st.stop()

st.title("Target Capture")
db_path = st.sidebar.text_input("Index database", CFG["db_path"])
store = Store(db_path)


@st.cache_resource
def tidy_index(path):
    """Once per start: drop rows left behind by videos removed while they were being indexed."""
    return Store(path).purge_orphans()


tidy_index(db_path)

def remove_media(media_id):
    """Remove one video from the index, with its cached pictures and the copies this app made
    (uploads, repaired DVR files). Videos you pointed to by path are never deleted."""
    path = store.remove_media(media_id)
    if os.path.isdir(THUMBS):
        for f in os.listdir(THUMBS):
            if f.startswith(f"{media_id}_"):
                os.remove(os.path.join(THUMBS, f))
    own = [os.path.abspath(UPLOADS), os.path.join(os.path.dirname(os.path.abspath(db_path)), "fixed_videos")]
    if path and os.path.isfile(path) and any(os.path.abspath(path).startswith(d + os.sep) for d in own):
        os.remove(path)
    live_dir = os.path.join(os.path.dirname(os.path.abspath(db_path)), "live_frames")
    if path and os.path.isdir(path) and os.path.abspath(path).startswith(live_dir + os.sep):
        shutil.rmtree(path, ignore_errors=True)  # pictures of a live camera day


@st.cache_resource
def app_version():
    return C.version()


@st.cache_resource
def device_line():
    from targetcapture.gpu import describe

    return describe()


with st.sidebar:
    st.caption(device_line())
    st.caption(f"Version {C.__version__}" + (f" · {app_version()['date']}" if app_version()["date"] else ""))
    st.subheader("Indexed media")
    for m in store.media():
        dur = fmt_offset(m["frames"] / m["fps"]) if m["kind"] == "video" else "image"
        row = st.columns([5, 1], vertical_alignment="center")
        row[0].caption(f"{os.path.basename(m['path'])} · {dur} · plates {'✓' if m['plates_done'] else '…'}"
                       f" · people {'✓' if m['persons_done'] else '…'}")
        with row[1].popover("", icon=":material/delete:", help="Remove this video from the index"):
            st.write(f"Remove **{os.path.basename(m['path'])}** and everything found in it?")
            if st.button("Remove", key=f"rm_{m['id']}", type="primary"):
                remove_media(m["id"])
                st.rerun()
    if store.media():
        with st.popover("Remove all videos", width="stretch"):
            st.write("Remove every video and all plates and people found, to start fresh? "
                     "Your original video files are not deleted, and what you taught in the Teach tab is kept.")
            if st.button("Remove all", type="primary"):
                for m in store.media():
                    remove_media(m["id"])
                shutil.rmtree(RESULTS, ignore_errors=True)
                st.rerun()


@st.cache_resource
def face_model():
    return P.FaceModel()


def show_rows(rows, out_dir, image_keys):
    for r in rows:
        with st.container(border=True):
            cols = st.columns([2, 1, 2])
            for k, label in (("frame_image", "Best capture"), ("second_frame_image", "Second capture")):
                if r.get(k):
                    cols[0].image(os.path.join(out_dir, r[k]), caption=label if r.get("second_frame_image") else None)
            for k in image_keys:
                if r.get(k):
                    cols[1].image(os.path.join(out_dir, r[k]))
            cols[2].json({k: v for k, v in r.items() if not k.endswith("_image") and v is not None}, expanded=True)


def downloads(rows, out_dir, title, key):
    """Excel and PDF buttons. The files are made only when a button is clicked, not with every search."""
    if not rows:
        return

    def excel():
        with open(to_excel(rows, out_dir, os.path.join(out_dir, "results.xlsx"), title), "rb") as f:
            return f.read()

    def pdf():
        with open(to_pdf(rows, out_dir, os.path.join(out_dir, "results.pdf"), title), "rb") as f:
            return f.read()

    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("Download Excel", excel, f"{key}.xlsx", key=f"xl_{key}", on_click="ignore",
                       icon=":material/table_view:")
    c2.download_button("Download PDF", pdf, f"{key}.pdf", key=f"pdf_{key}", on_click="ignore",
                       icon=":material/picture_as_pdf:")


def pager(n, key):
    """Page picker for long galleries; returns the slice to show."""
    pages = max(1, (n + PAGE - 1) // PAGE)
    page = st.number_input(f"Page (of {pages})", 1, pages, 1, key=key) if pages > 1 else 1
    return slice((page - 1) * PAGE, page * PAGE)


@st.cache_data(show_spinner="Grouping faces into people…")
def cached_identities(db, db_mtime, threshold):
    ppl = P.identities(Store(db), threshold=threshold)
    for p in ppl:
        p["embedding"] = p["embedding"].tolist()
        p["best"] = dict(p["best"])
        p["best"].pop("emb", None)
        for t in p["tracks"].values():
            t["best"] = dict(t["best"])
            t["best"].pop("emb", None)
    return ppl


def plate_data_sig(st_store) -> tuple:
    """A cheap fingerprint of the plate_reads table, used as the cache key for the grouped
    sightings below. It changes on any insert, delete, or in-place edit (a Teach correction
    rewrites `text`/`source`/`conf`), so the cache can never serve stale results."""
    r = st_store.db.execute(
        "SELECT COUNT(*), COALESCE(SUM(id), 0), COALESCE(SUM(LENGTH(text)), 0), "
        "COALESCE(SUM(LENGTH(COALESCE(source, ''))), 0), COALESCE(SUM(conf), 0) FROM plate_reads"
    ).fetchone()
    return tuple(r)


@st.cache_data(show_spinner="Grouping plate reads…")
def cached_plate_sightings(db, sig):
    """Grouped plate sightings, recomputed only when the plate data changes (see plate_data_sig).
    Repeated searches then reuse one grouping instead of re-grouping the whole index each time."""
    return L.group_sightings(Store(db).plate_reads())


@st.cache_data(show_spinner=False)
def cached_unread_captures(db, sig):
    """Grouped unread-plate captures, cached on the same fingerprint as the read sightings."""
    return L.group_unread(Store(db).plate_reads(unread=True))


@st.fragment(run_every=15)
def alert_banner():
    """New watch-list matches, refreshed every 15 s (live cameras add them while you look)."""
    new = Store(db_path).alerts(unseen_only=True)
    seen_before = st.session_state.setdefault("alerts_toasted", set())
    for a in new:
        if a["id"] not in seen_before:
            seen_before.add(a["id"])
            st.toast(f"Watch list: {a['kind']} {a['value']} found ({a['text']})", icon="🚨")
    if new:
        st.error(f"**{len(new)} new watch-list match(es).** Open the **Watch list** tab to see them.",
                 icon=":material/notifications_active:")


alert_banner()


def _live_media_ids(store):
    root = os.path.join(os.path.dirname(os.path.abspath(db_path)), "live_frames") + os.sep
    return {m["id"]: m for m in store.media() if m["path"].startswith(root)}


@st.fragment(run_every=2)
def live_view():
    """What the cameras see now, and the latest plates and faces they found (refreshes itself)."""
    store = Store(db_path)  # this runs in its own thread: its own connection
    now_dir = os.path.join(os.path.dirname(os.path.abspath(db_path)), "live_frames", "_now")
    cams = [c for c in store.cameras() if c["enabled"]]
    cols = st.columns(min(3, max(1, len(cams))))
    for i, c in enumerate(cams):
        p = os.path.join(now_dir, f"{c['id']}.jpg")
        with cols[i % len(cols)]:
            if os.path.exists(p):
                try:
                    with open(p, "rb") as f:
                        st.image(f.read(), caption=f"{c['name']} · {c['status'] or ''}", width="stretch")
                except OSError:
                    pass
            else:
                st.caption(f"{c['name']} · {c['status'] or 'starting'}")
    media = _live_media_ids(store)
    if not media:
        return
    ids = ",".join(str(i) for i in media)
    names = {m_id: os.path.basename(m["path"]).rsplit("_", 1)[0] for m_id, m in media.items()}
    alerts = [a for a in store.alerts(unseen_only=True) if a["media_id"] in media][:5]
    for a in alerts:
        st.error(f"Watch list: **{a['value']}** seen on {names[a['media_id']]} at "
                 f"{clock_time(media[a['media_id']], a['start'])} ({a['text']})", icon=":material/notifications_active:")
    left, right = st.columns(2)
    with left:
        st.markdown("**Latest plates**")
        rows = store.db.execute(f"SELECT * FROM plate_reads WHERE media_id IN ({ids}) AND text != '' "
                                "ORDER BY id DESC LIMIT 200").fetchall()
        seen, shown = set(), 0
        for r in rows:
            if r["text"] in seen or shown >= 8:
                continue
            seen.add(r["text"])
            shown += 1
            c1, c2 = st.columns([1, 2])
            th = thumbnail(r, store, THUMBS)
            if th:
                c1.image(th, width="stretch")
            c2.markdown(f"**{r['text']}**  \n{names[r['media_id']]} · {clock_time(media[r['media_id']], r['t'])[11:]}"
                        f"  \n{(r['vehicle_color'] or '')} {(r['vehicle_model'] or r['vehicle_type'] or '')}")
    with right:
        st.markdown("**Latest faces**")
        rows = store.db.execute(f"SELECT * FROM person_dets WHERE media_id IN ({ids}) AND fx1 IS NOT NULL "
                                "ORDER BY id DESC LIMIT 200").fetchall()
        seen, shown = set(), 0
        for r in rows:
            key = (r["media_id"], r["track_id"])
            if key in seen or shown >= 8:
                continue
            seen.add(key)
            shown += 1
            c1, c2 = st.columns([1, 2])
            th = thumbnail(r, store, THUMBS, box=("fx1", "fy1", "fx2", "fy2"), pad=0.3, min_w=96)
            if th:
                c1.image(th, width="stretch")
            c2.markdown(f"Person {r['track_id']}  \n{names[r['media_id']]} · "
                        f"{clock_time(media[r['media_id']], r['t'])[11:]}")

(tab_index, tab_dvr, tab_search, tab_watch, tab_cams, tab_faces, tab_all, tab_teach,
 tab_tips, tab_settings) = st.tabs(["Upload & index", "DVR footage", "Search",
                      "Watch list", "Live cameras", "All faces", "All plates", "Teach", "Best results", "Settings"])

with tab_index:
    up = st.file_uploader("Video or image", type=["mp4", "avi", "mkv", "mov", "dav", "h264", "ts",
                                                  "jpg", "jpeg", "png", "bmp", "webp"])
    path_in = st.text_input("…or a path to a file already on this machine (best for multi-hour DVR files)")
    c1, c2, c3 = st.columns(3)
    do_plates = c1.checkbox("Read plates", CFG["read_plates"])
    do_persons = c2.checkbox("Find people", CFG["find_people"])
    clock = c3.text_input("DVR clock at first frame (optional)", placeholder="2026-09-30 08:00:00")
    speeds = list(C.SPEED_NAMES)
    speed = st.radio("Speed", speeds, index=speeds.index(CFG["speed"]), format_func=C.SPEED_NAMES.get,
                     horizontal=True, help="Fast suits long DVR recordings. Thorough catches a few more plates on "
                                           "fast cars. Custom: set the frame rates under Advanced.")
    with st.expander("Advanced"):
        plate_fps = st.slider("Frames per second analysed for plates (Custom speed)", 1.0, 10.0,
                              float(CFG["plate_fps"]), disabled=speed != "custom")
        person_fps = st.slider("Frames per second analysed for people (Custom speed)", 1.0, 10.0,
                               float(CFG["person_fps"]), disabled=speed != "custom")
        plate_script = st.radio("Plate script", SCRIPTS, horizontal=True, index=SCRIPTS.index(CFG["plate_script"]),
                                format_func=lambda v: {"auto": "Auto (English and Arabic)",
                                                       "latin": "English only (fastest)",
                                                       "arabic": "Arabic / Kurdish only"}[v])
        car_models = st.checkbox("Recognise car make and model (Toyota Corolla, Nissan Altima...)", CFG["car_models"])
        vehicle_crops = st.checkbox("Also read plates inside each vehicle (slower; wide or 4K views)",
                                    CFG["vehicle_crops"])
        skip_empty = st.checkbox("Faster: skip plate reading when no vehicle is in view (quiet roads; "
                                 "may miss plates on steep top-down cameras)", CFG["skip_empty_road"])
        reindex = st.checkbox("Re-index from scratch")
    if st.button("Start indexing", type="primary", disabled=not (up or path_in)):
        if up:
            os.makedirs(UPLOADS, exist_ok=True)
            path = os.path.join(UPLOADS, up.name)
            with open(path, "wb") as f:
                shutil.copyfileobj(up, f, 16 << 20)
        else:
            path = path_in
        bars, began, shown = {}, {}, {}
        from targetcapture.video import probe

        _v = probe(path)
        vid_info = f"{_v.width}x{_v.height}, {_v.fps:.0f} frames/s"
        speed_line = st.empty()

        def progress(kind, i, total, found, t):
            import time

            now = time.time()
            if kind not in bars:
                bars[kind] = st.progress(0.0)
                began[kind] = (now, t)
            bars[kind].progress(min(1.0, i / max(1, total)),
                                text=f"{kind}: {fmt_offset(t)} analysed, {found} detections")
            t0, v0 = began[kind]
            if now - shown.get("at", 0) > 1 and now - t0 > 5 and t > v0:
                shown["at"] = now
                rate = (t - v0) / (now - t0)  # seconds of video per second
                left = max(0.0, (total / max(1, i)) * t - t) / rate
                from targetcapture.gpu import describe, gpu_memory_used
                from targetcapture.pipeline import live_timing, step_line

                mem = gpu_memory_used()
                speed_line.caption(
                    f"Speed: {rate:.1f}x real time (10 h of video in about {10 / rate:.1f} h). "
                    f"About {fmt_offset(left)} left.  \n{step_line(live_timing())}  \n"
                    f"{describe()}{f', graphics memory in use {mem}' if mem else ''}. "
                    f"Video: {vid_info}.")

        r = C.rates(CFG, speed) if speed != "custom" else {"plate_fps": plate_fps, "person_fps": person_fps,
                                                           "model_looks": 4}
        res = index_media(store, path, plates=do_plates, persons=do_persons, **r, clock_start=clock or None, vehicle_crops=vehicle_crops, plate_script=plate_script, car_models=car_models,
                          reindex=reindex, progress=progress, skip_empty_road=skip_empty,
                          min_ocr_conf=CFG["min_ocr_conf"])
        st.success(f"Indexed {os.path.basename(path)}. Now search it in the next tabs.")
        if res.get("timing") and res["info"].kind == "video":
            from targetcapture.pipeline import timing_lines

            lines = timing_lines(res["timing"], res["info"].duration)
            if lines:
                speed_line.empty()
                with st.expander("Indexing speed — where the time went (copy this if you want it tuned)",
                                 expanded=True):
                    st.code("\n".join(lines))
        if res.get("alerts"):
            st.error(f"{len(res['alerts'])} watch-list match(es) in this video: see the Watch list tab.")

def plate_table(sightings, key):
    """Plates as a table with a View tick box; returns the ticked sighting (or None)."""
    table = [{
        "plate": s["text"], "region / plate words": s["region"] or "",
        "start": fmt_offset(s["start"]), "end": fmt_offset(s["end"]),
        "clock": clock_time(store.media(s["media_id"]), s["start"]) or "", "reads": s["reads"],
        "conf": s["conf"], "vehicle": f"{s['vehicle_color'] or ''} {s['vehicle_model'] or s['vehicle_type'] or ''}".strip(),
        "source": {"taught": "typed by you", "learned": "learned"}.get(s["source"], "read"),
        "video": os.path.basename(store.media(s["media_id"])["path"]),
        "View": False,
    } for s in sightings]
    edited = st.data_editor(table, width="stretch", hide_index=True, key=f"{key}_{len(table)}",
                            disabled=[k for k in (table[0] if table else {}) if k != "View"],
                            column_config={"View": st.column_config.CheckboxColumn("View", width="small")})
    picked = [i for i, row in enumerate(edited) if row["View"]]
    return sightings[picked[-1]] if picked else None


def view_plate(s, key):
    out = os.path.join(RESULTS, "plate_view_" + "".join(ch for ch in s["text"] if ch.isalnum()))
    rows = plate_report(store, s["text"], [s], out)
    st.markdown(f"**{s['text']}** · {fmt_offset(s['start'])} to {fmt_offset(s['end'])}")
    downloads(rows, out, f"Plate {s['text']}", key)
    show_rows(rows, out, ["plate_image", "second_plate_image"])


def _search_plate():
    with st.form("plate_search"):
        q = st.text_input("Plate number: all of it or any part (21L1 finds 21L10415); wildcards: 12?BC*")
        c1, c2 = st.columns(2)
        max_dist = c1.number_input("Allowed character differences", 0, 3, int(CFG["plate_max_dist"]))
        min_reads = c2.number_input("Minimum reads per sighting", 1, 10, 1)
        if st.form_submit_button("Search plate") and q.strip():
            st.session_state.plate_query = (q.strip(), max_dist, min_reads)
    if st.session_state.get("plate_query"):
        q, max_dist, min_reads = st.session_state.plate_query
        hits = L.search(store, q, max_dist=max_dist, min_reads=min_reads,
                        sightings=cached_plate_sightings(db_path, plate_data_sig(store)))
        full = [h for h in hits if not h["partial"]]
        maybe = sorted((h for h in hits if h["partial"]), key=lambda h: (h["distance"], -h["reads"]))
        out = os.path.join(RESULTS, "plate_" + "".join(ch for ch in q if ch.isalnum()))
        rows = plate_report(store, q, full, out) if full else []
        st.subheader(f"{len(rows)} sighting(s) of {q}")
        if rows:
            st.caption(f"Full report saved to {os.path.join(out, 'report.html')}")
            downloads(rows, out, f"Plate search {q}", "plate_search")
            show_rows(rows, out, ["plate_image", "second_plate_image"])
        if maybe:
            st.subheader(f"Maybe the plate you want ({len(maybe)})")
            st.caption(f"Plates that contain **{q}**. Tick **View** to see one.")
            pick = plate_table(maybe, "plate_maybe")
            if pick:
                view_plate(pick, "plate_maybe_view")
        elif not rows:
            st.info("No plate found. Try fewer characters, or allow more differences.")

def _search_person():
    photos = st.file_uploader("Photo(s) of the person's face", type=["jpg", "jpeg", "png", "webp"],
                              accept_multiple_files=True)
    threshold = st.slider("Match threshold (higher = stricter)", 0.25, 0.7, 0.40, 0.01)
    if st.button("Search person", disabled=not photos):
        tmp = tempfile.mkdtemp()
        paths = []
        for p in photos:
            paths.append(os.path.join(tmp, p.name))
            with open(paths[-1], "wb") as f:
                f.write(p.getbuffer())
        try:
            embs = [face_model().query_embedding(p) for p in paths]
        except ValueError as e:
            st.error(str(e))
            st.stop()
        sightings, stats = P.search(store, embs, threshold=threshold)
        out = os.path.join(RESULTS, "person_" + os.path.splitext(photos[0].name)[0])
        rows = person_report(store, paths, sightings, stats, threshold, out)
        st.subheader(f"Seen {len(rows)} time(s)")
        st.caption(f"{stats.get('faces_indexed', 0)} faces in the index · {stats.get('near_misses', 0)} "
                   f"near misses just below the threshold · report: {os.path.join(out, 'report.html')}")
        downloads(rows, out, "Person search", "person_search")
        show_rows(rows, out, ["face_image"])

def _search_vehicle():
    st.caption("Every vehicle the plate indexing saw, also when its plate could not be read. "
               "Leave a filter empty to ignore it.")
    c1, c2, c3 = st.columns(3)
    words = c1.text_input("Make, model or words", placeholder="toyota camry")
    vtype = c2.selectbox("Type", ["any", "car", "truck", "bus", "motorcycle"])
    vcolor = c3.selectbox("Colour", ["any", "white", "black", "gray", "red", "blue", "green", "yellow",
                                     "orange", "brown", "purple"])
    c4, c5, c6 = st.columns(3)
    vplate = c4.selectbox("Plate", ["any", "read", "unread"],
                          format_func=lambda v: {"any": "any", "read": "plate was read",
                                                 "unread": "plate NOT read"}[v])
    t_from = c5.text_input("From", placeholder="00:10:00 or 2026-09-30 08:00:00")
    t_to = c6.text_input("To", placeholder="00:20:00 or 2026-09-30 09:00:00")
    hide_glimpse = st.checkbox("Hide vehicles seen only for a moment", True)
    if st.button("Search vehicles", type="primary"):
        found = V.search(store, words, None if vtype == "any" else vtype, None if vcolor == "any" else vcolor,
                         vplate, t_from or None, t_to or None, min_seconds=0.9 if hide_glimpse else 0)
        st.subheader(f"{len(found)} vehicle(s)")
        if len(found) > 200:
            st.caption("Showing the first 200: add filters to narrow it down.")
        out = os.path.join(RESULTS, "vehicle_search")
        shutil.rmtree(out, ignore_errors=True)
        rows = vehicle_report(store, found[:200], out)
        downloads(rows, out, "Vehicle search", "vehicle_search")
        show_rows(rows, out, ["vehicle_image"])


with tab_search:
    kind = st.radio("What are you looking for?", ["Plate", "Person", "Vehicle"], horizontal=True,
                    key="search_kind", label_visibility="collapsed")
    st.caption({"Plate": "Find a vehicle by its number plate, or any part of it.",
                "Person": "Find someone from a face photo across every indexed video.",
                "Vehicle": "Browse and filter every vehicle seen, by make, colour, type or time."}[kind])
    {"Plate": _search_plate, "Person": _search_person, "Vehicle": _search_vehicle}[kind]()

with tab_watch:
    st.caption("Wanted plates and people. Every new video and every live-camera minute is checked "
               "automatically; matches appear here and at the top of the page.")
    alerts = store.alerts()
    if alerts:
        h = st.columns([3, 1, 1])
        h[0].subheader(f"Matches ({sum(1 for a in alerts if not a['seen'])} new)")
        if h[1].button("Mark all as seen"):
            store.mark_alerts_seen()
            st.rerun()
        out = os.path.join(RESULTS, "alerts")
        rows = alert_report(store, alerts[:100], out)
        with h[2].popover("Download"):
            downloads(rows, out, "Watch-list matches", "watchlist_matches")
        show_rows(rows, out, ["plate_image", "face_image"])
    else:
        st.info("No matches yet.")
    st.subheader("Wanted")
    a1, a2 = st.columns(2)
    with a1.form("watch_plate", clear_on_submit=True):
        wp = st.text_input("Plate number (wildcards allowed: 21?45194, 21*)")
        wn = st.text_input("Note (optional)", key="wp_note")
        if st.form_submit_button("Add plate") and wp:
            try:
                W.add_plate(store, wp, wn)
                st.success(f"Added {wp}. Checking the videos already indexed…")
                W.check(store)
            except ValueError as e:
                st.error(str(e))
    with a2.form("watch_face", clear_on_submit=True):
        wf = st.file_uploader("Photo of the person's face", type=["jpg", "jpeg", "png", "webp"])
        wfn = st.text_input("Name or note (optional)", key="wf_note")
        if st.form_submit_button("Add person") and wf:
            tmp = os.path.join(tempfile.mkdtemp(), wf.name)
            with open(tmp, "wb") as f:
                f.write(wf.getbuffer())
            try:
                W.add_face(store, tmp, face_model(), wfn)
                st.success("Added. Checking the videos already indexed…")
                W.check(store)
            except ValueError as e:
                st.error(str(e))
    for w in store.watchlist():
        row = st.columns([1, 4, 1])
        if w["photo"]:
            row[0].image(w["photo"], width=70)
        else:
            row[0].markdown(f"### {w['value']}")
        row[1].markdown(f"**{w['kind']}: {w['value']}**  \n{w['note'] or ''}")
        if row[2].button("Remove", key=f"wrm_{w['id']}"):
            store.remove_watch(w["id"])
            st.rerun()

STOP_FILE = os.path.join(os.path.dirname(os.path.abspath(db_path)), "live.stop")


def live_running(st_store) -> bool:
    """A camera reported in the last 90 s (the live process writes every few seconds)."""
    now = dt.datetime.now()
    return any(c["last_seen"] and (now - dt.datetime.fromisoformat(c["last_seen"])).total_seconds() < 90
               for c in st_store.cameras())


def start_live():
    """Start live monitoring as a hidden background process (it keeps running when the page closes)."""
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    subprocess.Popen([sys.executable, "-m", "targetcapture", "--db", db_path, "live", "--stop-file", STOP_FILE],
                     creationflags=flags, cwd=APP_DIR, stdout=open(os.path.join(APP_DIR, "live.log"), "a"),
                     stderr=subprocess.STDOUT)


@st.cache_resource
def _autostart_once():
    """Settings: start live monitoring when the page starts (once per page start, not per browser tab)."""
    return {"done": False}


_auto = _autostart_once()
if not _auto["done"]:
    _auto["done"] = True
    if CFG["live_autostart"] and store.cameras() and not live_running(store):
        start_live()
        st.toast("Live monitoring started (Settings: start when the page opens).")


def folder_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024


def indexed_downloads():
    """Downloaded DVR videos that are already indexed (safe to delete: results keep their pictures)."""
    from targetcapture.dvr import default_folder

    folders = {os.path.abspath(f) + os.sep for f in (default_folder(db_path), CFG["dvr_folder"]) if f}
    out = []
    for m in store.db.execute("SELECT path FROM media").fetchall():
        p = os.path.abspath(m["path"] or "")
        if any(p.startswith(f) for f in folders) and os.path.isfile(p):
            out.append(p)
    return out


def delete_downloads_button(key):
    files = indexed_downloads()
    size = sum(os.path.getsize(f) for f in files)
    with st.popover(f"Delete indexed DVR downloads ({human(size)})", disabled=not files,
                    help="Videos downloaded from the DVR that are already indexed. Results and their pictures stay."):
        st.write(f"Delete {len(files)} downloaded video(s), {human(size)}? Search results stay, with their pictures.")
        if st.button("Yes, delete", key=key, type="primary"):
            for f in files:
                try:
                    os.remove(f)
                except OSError:
                    pass
            st.rerun()


def start_fetch(job_path):
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    subprocess.Popen([sys.executable, "-m", "targetcapture", "--db", db_path, "fetch", job_path], creationflags=flags,
                     cwd=APP_DIR, stdout=open(os.path.join(APP_DIR, "dvr_fetch.log"), "a"), stderr=subprocess.STDOUT)


def dvr_defaults():
    """DVR address and login from a camera added in the Live cameras tab (so they are typed once)."""
    from urllib.parse import unquote, urlparse

    for c in store.cameras():
        u = urlparse(c["url"])
        if u.scheme == "rtsp" and u.hostname:
            make = "Dahua / Imou" if "realmonitor" in (u.path + "?" + u.query) else "Hikvision / HiLook"
            return {"ip": u.hostname, "rtsp_port": u.port or 554, "user": unquote(u.username or "admin"),
                    "password": unquote(u.password or ""), "make": make, "name": c["name"].rsplit(" ch", 1)[0]}
    return {"ip": "", "rtsp_port": 554, "user": "admin", "password": "", "make": "Hikvision / HiLook", "name": "DVR"}


def known_dvrs(found: list[dict], cameras_too: bool = False) -> dict:
    """DVRs / NVRs found on the network, saved in this tab or added in Live cameras (one per address)."""
    from urllib.parse import unquote, urlparse

    out = {d["name"]: {**dict(d), "source": "saved"} for d in store.dvrs()}
    ips = {d["ip"] for d in out.values()}
    for c in store.cameras():
        u = urlparse(c["url"])
        if u.scheme != "rtsp" or not u.hostname or u.hostname in ips or " ch" not in c["name"]:
            continue
        ips.add(u.hostname)
        name = c["name"].rsplit(" ch", 1)[0]
        out.setdefault(name if name not in out else f"{name} ({u.hostname})", {
            "name": name, "ip": u.hostname, "user": unquote(u.username or "admin"),
            "password": unquote(u.password or ""), "http_port": 80, "rtsp_port": u.port or 554,
            "make": "Dahua / Imou" if "realmonitor" in (u.path + "?" + u.query) else "Hikvision / HiLook",
            "source": "Live cameras"})
    by_ip = {d["ip"]: d for d in out.values()}
    for f in found:
        if f["ip"] in by_ip:
            by_ip[f["ip"]]["net"] = f
        elif f["kind"] == "DVR / NVR" or cameras_too:
            name = f"{f['model'] or f['make']} ({f['ip']})"
            out[name] = {"name": f["model"] or "DVR", "ip": f["ip"], "user": "admin", "password": "",
                         "http_port": f["http_port"], "rtsp_port": 554, "make": f["make"],
                         "source": "found on network", "net": f}
    return dict(sorted(out.items(), key=lambda nd: "net" not in nd[1]))  # the ones on the network first


def live_stream_path(make, ip, user, password, port, ch):
    """Which stream address form this DVR / NVR answers for channel `ch` (the same checks as
    "Test channel 1 and add all channels"): (channel -> path or None, RTSP port, problem, notes)."""
    from targetcapture.live import check_login, dvr_paths, hik_rtsp_port, rtsp_describe

    notes = []
    login, info = check_login(make, ip, user, password)
    if login is False:
        return None, port, f"the DVR's web login says: {info}.", notes
    if login:
        notes.append(f"Login is correct{' (' + info + ')' if info else ''}.")
        set_port = hik_rtsp_port(ip, user, password) if make.lower().startswith("hik") else None
        if set_port and set_port != port:
            notes.append(f"The DVR's RTSP port is {set_port}, using that.")
            port = set_port
    for label, path in dvr_paths(make):
        code, note = rtsp_describe(ip, port, path(ch), user, password)
        notes.append(f"{label} ({path(ch)}): " + {200: "found", 401: "login refused", 404: "not on this DVR"}
                     .get(code, f"no answer ({note})" if code is None else f"answer {code}"))
        if code == 200:
            return path, port, "", notes
        if code is None:
            return None, port, (f"no answer on port {port}: check that this PC is on the same network as the "
                                "DVR, and the RTSP port in the DVR's network settings."), notes
        if code == 401:  # same login on every path: do not lock the DVR with retries
            return None, port, (note or "the DVR refuses this user name and password for video") + ".", notes
    return None, port, "the DVR did not accept any of the usual stream addresses.", notes


def live_dvr_picker(cams):
    """Live cameras tab: pick a DVR / NVR, then its cameras, and start watching them live."""
    from targetcapture import dvr as D

    st.subheader("Watch cameras of a DVR / NVR")
    if "live_dvr_msg" in st.session_state:
        st.success(st.session_state.pop("live_dvr_msg"))
    known_ = known_dvrs(st.session_state.get("dvr_found", []), st.session_state.get("dvr_cams_too", False))
    if not known_:
        st.info("No DVR / NVR found on this PC's network. Click **Search the network again** in the DVR footage "
                "tab, or type its address under *Type a DVR / NVR address by hand* below.")
        return
    cams_seen = st.session_state.setdefault("dvr_cams", {})  # shared with the DVR footage tab
    logins = st.session_state.setdefault("dvr_logins", {})   # typed this visit, not saved
    names = list(known_)
    if st.session_state.get("live_dvr_pick") not in names:
        st.session_state["live_dvr_pick"] = names[0]

    def pick_dvr():
        sel = st.session_state["live_dvr_list"].selection.rows
        if sel:
            st.session_state["live_dvr_pick"] = names[sel[0]]

    def row(n, d):
        chans, model = cams_seen.get(at(d), (None, ""))
        net = d.get("net") or {}
        return {"DVR / NVR": n, "make": d["make"], "IP address": d["ip"],
                "model": (model if chans is not None else "") or net.get("model", ""),
                "cameras": f"{len(chans)} ({sum(c['status'] == 'online' for c in chans)} online)"
                if chans is not None else ("type the password" if not (d["password"] or at(d) in logins) else ""),
                "on network": "yes" if net else ""}

    st.caption("1. Click a DVR / NVR.")
    st.dataframe([row(n, d) for n, d in known_.items()], hide_index=True, width="stretch",
                 on_select=pick_dvr, selection_mode="single-row", key="live_dvr_list")
    pick = st.session_state["live_dvr_pick"]
    d = known_[pick]
    key = at(d)
    user, pw = logins.get(key, (d["user"] or "admin", d["password"]))
    if not pw:
        with st.form("live_dvr_login"):
            st.markdown(f"**{pick}**: type its login to see its cameras.")
            c1, c2 = st.columns(2)
            u = c1.text_input("User name", user, key=f"live_u_{key}")
            p = c2.text_input("Password", type="password", key=f"live_p_{key}")
            save = st.checkbox("Save this DVR's login for next time", True, key=f"live_save_{key}")
            if st.form_submit_button("Show its cameras", type="primary") and p:
                with st.spinner(f"Asking {pick} for its cameras…"):
                    cams_seen[key] = D.list_channels(d["make"], d["ip"], u, p, int(d["http_port"] or 80))
                if cams_seen[key][0] is None:
                    st.error(f"Could not read the camera list: {cams_seen.pop(key)[1]}.")
                else:
                    logins[key] = (u, p)
                    if save:
                        name = d["name"] if d["source"] == "saved" else pick
                        store.save_dvr(name, d["make"], d["ip"], u, p, int(d["http_port"] or 80),
                                       int(d["rtsp_port"] or 554))
                        st.session_state["live_dvr_pick"] = name
                    st.rerun()
        return
    if key not in cams_seen:
        with st.spinner(f"Asking {pick} for its cameras…"):
            cams_seen[key] = D.list_channels(d["make"], d["ip"], user, pw, int(d["http_port"] or 80))
    chans, model = cams_seen[key]
    if chans is None:
        st.error(f"Could not read {pick}'s cameras: {model}.")
        if st.button("Ask again", key="live_dvr_retry"):
            cams_seen.pop(key, None)
            st.rerun()
        return
    if not chans:
        st.info(f"{pick} answered but has no cameras / channels set up.")
        return
    base = d["name"] if d["source"] in ("saved", "Live cameras") else pick
    have = {c["name"]: c for c in cams}
    rows = [{"channel": c["channel"], "name": c["name"], "status": c["status"] or "unknown",
             "type": "IP camera" if c["kind"] == "IP" else "analog", "IP address": c["ip"], "model": c["model"],
             "live": ("watching" if have[f"{base} ch{c['channel']}"]["enabled"] else "added, switched off")
             if f"{base} ch{c['channel']}" in have else ""} for c in chans]
    st.caption(f"2. Select the cameras of **{pick}** to watch (click rows; Ctrl or Shift for several).")
    ev = st.dataframe(rows, hide_index=True, width="stretch", on_select="rerun", selection_mode="multi-row",
                      key=f"live_cam_table_{key}",
                      column_config={"channel": st.column_config.NumberColumn(width="small")})
    sel = [chans[i] for i in ev.selection.rows]
    st.caption(f"3. The login used for the video of **{pick}**" + (" (saved)." if d["password"] else ".")
               + " Change it here if it is not right.")
    l1, l2, l3 = st.columns([1, 1, 2])
    user = l1.text_input("User name", user, key=f"live_vu_{key}")
    pw = l2.text_input("Password", pw, type="password", key=f"live_vp_{key}")
    k1, k2, k3 = st.columns([1, 1, 2])
    lp = k1.checkbox("Read plates", CFG["read_plates"], key="live_dvr_p")
    lpp = k2.checkbox("Find people", CFG["find_people"], key="live_dvr_pp")
    go = k3.button("4. Start live indexing on the selected cameras", type="primary", disabled=not (sel and pw),
                   key="live_dvr_go")
    if sel:
        k3.caption(f"{len(sel)} selected: " + ", ".join(f"{c['channel']}" + (f" · {c['name']}" if c["name"] else "")
                                                      for c in sel))
    if "live_dvr_pic" in st.session_state:
        jpg, caption = st.session_state.pop("live_dvr_pic")
        st.image(jpg, caption=caption, width=480)
    if not go:
        return
    from targetcapture.live import stream_url, sub_stream, test_camera

    first = sel[0]["channel"]
    notes, ok, jpg, how = [], False, None, ""
    with st.status(f"Connecting to channel {first}…", expanded=True) as box:
        path, port, problem, steps = live_stream_path(d["make"], d["ip"], user, pw, int(d["rtsp_port"] or 554), first)
        notes += steps
        for n_ in steps:
            st.write(n_)
        if path:
            # The stream address answering is not enough: get a real picture, as the hand-typed form does.
            # Many DVRs send no usable picture on the main stream (H.265+), but do on the sub stream.
            tries = [("main stream", path, "tcp", 20), ("main stream, read by Target Capture itself", path, "own", 20),
                     ("sub stream", lambda ch, main=path: sub_stream(main(ch)), "tcp", 15),
                     ("main stream over UDP", path, "udp", 15)]
            for how, p_, transport, secs in tries:
                if how == "sub stream" and p_(first) == path(first):
                    continue
                st.write(f"Getting a picture from the {how} (up to {secs} s)…")
                ok, msg, jpg = test_camera(stream_url(d["ip"], port, p_(first), user, pw), secs, transport,
                                           http_port=int(d["http_port"] or 80))
                notes.append(f"{how} ({p_(first)}): {msg}")
                st.write(msg)
                if ok:
                    path = p_
                    break
        box.update(label=f"Got a picture ({how})" if ok else "Could not connect", state="complete" if ok else "error")
    if not path or not ok:
        if not path:
            st.error(f"Could not connect to channel {first}: {problem}")
        else:
            st.error(f"Channel {first} answers, but sent no picture on the main stream, the sub stream or over UDP. "
                     "Check that a camera is connected to it, and in the DVR's Video settings set the video type to "
                     "H.264 instead of H.265 / H.265+. Nothing was added.")
        with st.expander("Details (send these if you ask for help)"):
            st.code("\n".join(notes))
        return
    logins[key] = (user, pw)
    if d["source"] == "saved" and (user, pw) != (d["user"], d["password"]):
        store.save_dvr(d["name"], d["make"], d["ip"], user, pw, int(d["http_port"] or 80), port)
    for c in sel:
        name = f"{base} ch{c['channel']}"
        url = stream_url(d["ip"], port, path(c["channel"]), user, pw)
        if name in have:
            store.update_camera(have[name]["id"], url=url, enabled=1, plates=int(lp), persons=int(lpp))
        else:
            store.add_camera(name, url, lp, lpp)
    started = not live_running(store)
    if started:
        start_live()
    st.session_state["live_dvr_pic"] = (jpg, f"Channel {first} ({how})")
    st.session_state["live_dvr_msg"] = (
        f"Watching {len(sel)} camera(s) of {pick} live" + (f", using the {how}" if how != "main stream" else "") + ". "
        + ("Live monitoring is starting: " if started else "")
        + "each camera shows 'recording' within a minute. Every camera uses part of the graphics card.")
    st.rerun()


def dvr_cameras(found, picker_key, title):
    """The DVR's own camera list; picking a row points the footage download at that channel."""
    if not found:
        return
    chans, model = found
    st.subheader(f"Cameras on {title}" + (f" ({model})" if chans is not None and model else ""))
    if chans is None:
        st.error(f"Could not read the camera list: {model}. Check the address, web port and login below, "
                 "then click Read the cameras again.")
        return
    if not chans:
        st.info("The DVR answered but has no cameras / channels set up.")
        return
    on = sum(c["status"] == "online" for c in chans)
    st.caption(f"{len(chans)} channel(s), {on} online. Click a camera to see its details and download only its "
               "channel below.")
    rows = [{"channel": c["channel"], "name": c["name"], "status": c["status"] or "unknown", "type": c["kind"],
             "IP address": c["ip"], "port": c["port"], "model": c["model"], "protocol": c["protocol"],
             "resolution": c["resolution"]} for c in chans]
    table_key = "dvr_cam_table_" + picker_key

    def focus():
        sel = st.session_state[table_key].selection.rows
        if sel:
            st.session_state[picker_key] = [chans[sel[0]]["channel"]]

    ev = st.dataframe(rows, hide_index=True, width="stretch", on_select=focus, selection_mode="single-row",
                      key=table_key, column_config={"channel": st.column_config.NumberColumn(width="small")})
    if ev.selection.rows:
        c = chans[ev.selection.rows[0]]
        st.markdown(f"**Channel {c['channel']} · {c['name'] or 'no name'}**: {c['status'] or 'status unknown'}")
        facts = {"Type": "IP camera" if c["kind"] == "IP" else "analog camera", "IP address": c["ip"],
                 "Port": c["port"], "Protocol": c["protocol"] if c["kind"] == "IP" else "",
                 "Video format": c["protocol"] if c["kind"] != "IP" else "", "Model": c["model"],
                 "Serial number": c["serial"], "Firmware": c["firmware"], "Resolution": c["resolution"]}
        st.caption(" · ".join(f"{k_}: {v}" for k_, v in facts.items() if v))
        st.caption("The Channels box below now holds only this channel: pick the time and click "
                   "Check what is recorded or Download and index.")


with tab_dvr:
    from targetcapture import dvr as D

    st.caption("Pick a DVR / NVR, the channels and a time: the recordings are downloaded (several pieces at "
               "once) and each piece is indexed with your Settings as soon as it arrives. It keeps running when "
               "this page is closed. Results appear in the search tabs after each piece, with the DVR's own clock "
               "time.")
    from targetcapture.live import clean_host

    from targetcapture import discover as F

    NEW = "New DVR / NVR"
    if "dvr_found" not in st.session_state:  # once per visit; "Search the network again" repeats it
        with st.spinner("Looking for DVRs / NVRs on your network…"):
            st.session_state["dvr_found"] = F.search(3)
    found = st.session_state["dvr_found"]
    known = known_dvrs(found, st.session_state.get("dvr_cams_too", False))
    cams_seen = st.session_state.setdefault("dvr_cams", {})  # "ip:port" -> (channels or None, model or reason)

    def at(d):
        return f"{d['ip']}:{int(d['http_port'] or 80)}"
    if st.session_state.get("dvr_pick") not in [NEW, *known]:
        st.session_state["dvr_pick"] = next(iter(known), NEW)
    pick = st.session_state["dvr_pick"]

    st.subheader("Your DVRs / NVRs")
    if known:
        names = list(known)

        def pick_dvr():
            sel = st.session_state["dvr_list"].selection.rows
            if sel:
                st.session_state["dvr_pick"] = names[sel[0]]

        def row(n, d):
            chans, model = cams_seen.get(at(d), (None, ""))
            ok, net = chans is not None, d.get("net") or {}
            return {"name": n, "make": d["make"], "IP address": d["ip"],
                    "model": (model if ok else "") or net.get("model", ""),
                    "cameras": f"{len(chans)} ({sum(c['status'] == 'online' for c in chans)} online)" if ok
                    else ("could not read" if at(d) in cams_seen else
                          ("type the password" if not d["password"] else "")),
                    "on network": "yes" if net else ("not found" if found else ""),
                    "from": d["source"], "serial / MAC": net.get("serial") or net.get("mac", ""),
                    "note": net.get("note", "")}

        st.dataframe([row(n, d) for n, d in known.items()],
                     hide_index=True, width="stretch", on_select=pick_dvr, selection_mode="single-row",
                     key="dvr_list")
        st.caption(f"{len(found)} device(s) answered on your network (Hikvision SADP, Dahua and ONVIF search). "
                   "Click a DVR / NVR to see its cameras; a new one asks for its password first.")
    else:
        st.info("No DVR / NVR answered on this PC's network. Check that this PC is on the same network as the "
                "DVR, or type its address below.")
    n1, n2 = st.columns([1, 3])
    if n1.button("Search the network again", key="dvr_search"):
        st.session_state.pop("dvr_found", None)
        st.rerun()
    n2.checkbox("Also show single IP cameras", key="dvr_cams_too")
    s1, s2, s3 = st.columns([2, 1, 1])
    if pick in known and s2.button("Read the cameras again", key="dvr_reread"):
        cams_seen.pop(at(known[pick]), None)
    if s3.button("Add a new DVR / NVR", key="dvr_new"):
        st.session_state["dvr_pick"] = pick = NEW
    if pick in known and known[pick]["source"] == "saved" and s1.button(f"Forget {pick}", key="dvr_forget"):
        store.remove_dvr(pick)
        st.session_state.pop("dvr_pick", None)
        st.rerun()

    if pick in known:
        dd = known[pick]
        if not dd["password"]:
            st.info(f"{pick} is on your network. Type its user name and password below and click "
                    "**List the DVR's cameras**.")
        elif at(dd) not in cams_seen:
            with st.spinner(f"Asking {pick} for its cameras…"):
                cams_seen[at(dd)] = D.list_channels(dd["make"], dd["ip"], dd["user"], dd["password"],
                                                      int(dd["http_port"] or 80))
            st.rerun()  # the DVR list shows the camera count
    else:
        dd = {**dvr_defaults(), "name": "DVR", "ip": "", "password": "", "http_port": 80}
    k = f"_{pick}"  # the fields refresh when another DVR is picked
    makes = ["Hikvision / HiLook", "Dahua / Imou"]
    typed_at = at({"ip": clean_host(st.session_state.get("di" + k, dd["ip"]))[0].strip(),
                   "http_port": st.session_state.get("dh" + k, dd.get("http_port") or 80)})
    ch_names = {c["channel"]: c["name"] for c in (cams_seen.get(typed_at) or (None, ""))[0] or [] if c["name"]}
    st.session_state.setdefault("dc" + k, [1])
    if pick in known or typed_at in cams_seen:
        dvr_cameras(cams_seen.get(typed_at), "dc" + k, pick if pick in known else typed_at.split(":")[0])
    st.subheader(f"Download footage from {pick}" if pick in known else "New DVR / NVR")
    with st.form("dvr_fetch"):
        a1, a2, a3, a4 = st.columns(4)
        f_name = a1.text_input("Name", dd["name"], key="dn" + k)
        f_ip = a2.text_input("IP address", dd["ip"], placeholder="192.168.1.230", key="di" + k)
        f_make = a3.selectbox("Make", makes, index=makes.index(dd["make"]) if dd["make"] in makes else 0, key="dm" + k)
        f_ch = a4.multiselect("Channels", list(range(1, 65)), key="dc" + k,
                              format_func=lambda c: f"{c} · {ch_names[c]}" if c in ch_names else str(c))
        b1, b2, b3, b4 = st.columns(4)
        f_user = b1.text_input("User name", dd["user"], key="du" + k)
        f_pass = b2.text_input("Password", dd["password"], type="password", key="dp" + k)
        f_http = b3.number_input("Web port", 1, 65535, int(dd.get("http_port") or 80), key="dh" + k,
                                 help="The port of the DVR's web page (normally 80).")
        f_rtsp = b4.number_input("RTSP port", 1, 65535, int(dd["rtsp_port"]), key="dr" + k)
        today = dt.date.today()
        c1, c2, c3, c4 = st.columns(4)
        d_from = c1.date_input("From date", today)
        t_from = c2.time_input("From time", dt.time(8, 0), step=60)
        d_to = c3.date_input("To date", today)
        t_to = c4.time_input("To time", dt.time(9, 0), step=60)
        e1, e2, e3, e4 = st.columns(4)
        f_plates = e1.checkbox("Read plates", CFG["read_plates"])
        f_people = e1.checkbox("Find people", CFG["find_people"])
        speeds = [s_ for s_ in C.SPEED_NAMES if s_ != "custom"]
        f_speed = e2.radio("Speed", speeds, format_func=C.SPEED_NAMES.get,
                           index=speeds.index(CFG["speed"]) if CFG["speed"] in speeds else 0)
        f_piece = e3.selectbox("Piece length", [5, 10, 15, 30, 60], index=1, format_func=lambda m: f"{m} minutes",
                               help="Shorter pieces show the first results sooner.")
        f_par = e3.selectbox("Download at the same time", [1, 2, 3, 4], index=min(3, int(CFG["dvr_parallel"]) - 1),
                             format_func=lambda n: f"{n} piece{'s' if n > 1 else ''}",
                             help="The DVR sends each download at a limited speed; several at once is faster "
                                  "in total. Lower it if the DVR's live view stutters.")
        f_method = e4.selectbox("Download by", ["auto", "web", "rtsp"], format_func={
            "auto": "Fastest that works", "web": "Web download only (fast)",
            "rtsp": "Playback only (real-time speed)"}.get)
        f_delete = e4.checkbox("Delete each video after indexing", CFG["dvr_delete_after"],
                               help="Saves disk space. The pictures of plates and faces are kept, so results "
                                    "still show; only playing the video is no longer possible.")
        f_folder = st.text_input("Save downloads in", CFG["dvr_folder"] or D.default_folder(db_path),
                                 help="Any folder on this PC or a USB / network drive, e.g. D:\\DVR downloads")
        f_save = st.checkbox("Save these DVR details (name, address, login) for next time",
                             pick not in known or known[pick]["source"] != "saved")
        g1, g2, g3 = st.columns([1, 1, 1])
        list_cams = g1.form_submit_button("List the DVR's cameras",
                                          help="Read the cameras / channels set up in the DVR / NVR itself.")
        check = g2.form_submit_button("Check what is recorded")
        go = g3.form_submit_button("Download and index", type="primary")
    start_t, end_t = dt.datetime.combine(d_from, t_from), dt.datetime.combine(d_to, t_to)
    f_ip, typed_port = clean_host(f_ip)  # also takes a pasted web address like http://192.168.1.230/
    if typed_port and typed_port != int(f_rtsp):  # a web-page address with its port
        f_http = typed_port
    if list_cams:
        if not f_ip.strip() or not f_pass:
            st.error("Type the IP address and password.")
        else:
            with st.spinner("Asking the DVR for its cameras…"):
                chans, why = D.list_channels(f_make, f_ip.strip(), f_user, f_pass, int(f_http))
            if chans is None:
                st.error(f"Could not read the camera list: {why}.")
            else:
                cams_seen[at({"ip": f_ip.strip(), "http_port": f_http})] = (chans, why)
                if f_save:
                    store.save_dvr(f_name.strip() or "DVR", f_make, f_ip, f_user, f_pass, f_http, f_rtsp)
                    st.session_state["dvr_pick"] = f_name.strip() or "DVR"
                st.rerun()  # the channel picker shows the camera names
    if (check or go) and (not f_ip.strip() or not f_pass or not f_ch or end_t <= start_t):
        st.error("Type the IP address and password, pick at least one channel, and an end after the start.")
    elif check or go:
        if f_save:
            store.save_dvr(f_name.strip() or "DVR", f_make, f_ip, f_user, f_pass, f_http, f_rtsp)
        folder = f_folder.strip() or D.default_folder(db_path)
        if go:  # remember the download choices
            C.save({**C.load(), "dvr_folder": "" if folder == D.default_folder(db_path) else folder,
                    "dvr_delete_after": f_delete, "dvr_parallel": f_par})
    if check and f_ip and f_pass and f_ch and end_t > start_t:
        with st.spinner("Asking the DVR…"):
            for ch in f_ch:
                rec = D.find_recordings(f_make, f_ip.strip(), f_user, f_pass, ch, start_t, end_t, int(f_http))
                if rec is None:
                    st.warning(f"Channel {ch}: the DVR did not answer the recording search (web port "
                               f"{f_http}, user and password). Download and index still tries every piece.")
                else:
                    ps = D.pieces(start_t, end_t, 1, rec)
                    mins = len(ps)
                    spans = ", ".join(f"{a:%H:%M}–{b:%H:%M}" for a, b in D.pieces(start_t, end_t, 24 * 60, rec)[:12])
                    (st.success if mins else st.info)(
                        f"Channel {ch}: {mins} minute(s) recorded" + (f" ({spans})" if spans else "") + ".")
    elif go and f_ip and f_pass and f_ch and end_t > start_t:
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as e:
            st.error(f"Cannot use the folder {folder}: {e}")
        else:
            job = {"name": f_name.strip() or "DVR", "make": f_make, "ip": f_ip.strip(), "user": f_user,
                   "password": f_pass, "http_port": int(f_http), "rtsp_port": int(f_rtsp), "channels": f_ch,
                   "start": start_t.isoformat(sep=" "), "end": end_t.isoformat(sep=" "), "piece_minutes": f_piece,
                   "method": f_method, "plates": f_plates, "persons": f_people, "speed": f_speed,
                   "delete_after": f_delete, "folder": folder, "parallel": f_par}
            start_fetch(D.new_job(db_path, job))
            st.success("Started. Progress shows below.")

    delete_downloads_button("del_dl_tab")
    with st.expander("Index videos downloaded with the DVR's own web page"):
        st.caption("If you download with the DVR / NVR's web page, save the files into one folder and start this: "
                   "every video in it is indexed, and with 'Keep watching' each new file is indexed as soon as "
                   "its download finishes. The time in the file name (e.g. ..._20261014100000_...) is used as the "
                   "clock time.")
        w1, w2 = st.columns([3, 1])
        w_folder = w1.text_input("Folder", CFG["dvr_folder"] or D.default_folder(db_path), key="watch_folder")
        w_watch = w2.checkbox("Keep watching for new files", True)
        w_delete = w2.checkbox("Delete each video after indexing", CFG["dvr_delete_after"], key="watch_del")
        w3, w4, w5 = st.columns(3)
        w_plates = w3.checkbox("Read plates", CFG["read_plates"], key="watch_plates")
        w_people = w4.checkbox("Find people (faces)", CFG["find_people"], key="watch_people",
                               help="Off = plates only, which is faster.")
        w_speeds = [s_ for s_ in C.SPEED_NAMES if s_ != "custom"]
        w_speed = w5.selectbox("Speed", w_speeds, format_func=C.SPEED_NAMES.get, key="watch_speed",
                               index=w_speeds.index(CFG["speed"]) if CFG["speed"] in w_speeds else 0)
        if st.button("Index this folder"):
            if not (w_plates or w_people):
                st.error("Tick plates, people or both.")
            elif not os.path.isdir(w_folder.strip()):
                st.error("That folder does not exist.")
            else:
                job = {"kind": "folder", "name": os.path.basename(os.path.normpath(w_folder.strip())) or "folder",
                       "folder": w_folder.strip(), "watch": w_watch, "delete_after": w_delete,
                       "plates": w_plates, "persons": w_people, "speed": w_speed}
                start_fetch(D.new_job(db_path, job))
                st.success("Started. Progress shows below.")

    @st.fragment(run_every=3)
    def dvr_jobs():
        jobs = D.list_jobs(db_path)
        if not jobs:
            return
        st.subheader("Downloads")
        for j in jobs[:6]:
            ps = j.get("pieces", [])
            done = sum(p["state"] == "done" for p in ps)
            running = j["state"] in ("running", "finding recordings")
            stale = D.is_stale(j)
            if j.get("kind") == "folder":
                what = f"**Folder {j['folder']}** · {done} file(s) indexed · "
            else:
                what = (f"**{j['name']}** · channel(s) {', '.join(map(str, j['channels']))} · {j['from'][:16]} to "
                        f"{j['to'][:16]} · {done} of {len(ps)} pieces indexed · ")
            h1, h2 = st.columns([5, 1])
            h1.markdown(what + ("stopped unexpectedly" if stale else j["state"]))
            if running and not stale and h2.button("Stop", key=f"stop_{j['id']}"):
                open(j["_path"].replace(".status.json", ".stop"), "w").close()
            if (not running or stale) and h2.button("Clear", key=f"clear_{j['id']}",
                                                     help="Remove this line (indexed results stay)."):
                D.remove_job(j["_path"])
                st.rerun()
            if ps and j.get("kind") != "folder":
                h1.progress(done / len(ps))
            if j.get("download"):
                h1.caption(j["download"])
            if j.get("indexing") and running and not stale:
                h1.caption(j["indexing"])
            for n in j.get("notes", []):
                h1.caption(n)
            if stale:
                h1.caption("The download program was closed (window closed, PC restarted or an error, see "
                           "dvr_fetch.log). Clear this line and start the download again for the pieces not indexed.")
            with st.expander("Pieces" if j.get("kind") != "folder" else "Files", expanded=running and not stale):
                rows = [{**({"file": p.get("file")} if j.get("kind") == "folder" else
                            {"channel": p["channel"], "from": p["start"][11:16], "to": p["end"][11:16]}),
                         "state": p["state"],
                         "found": ", ".join(f"{v} {k}" for k, v in p.get("found", {}).items()),
                         "speed": p.get("indexed_in", ""), "note": p.get("note", "")} for p in ps]
                st.dataframe(rows, hide_index=True, width="stretch")

    dvr_jobs()


with tab_cams:
    st.caption("Connect DVR / NVR / IP cameras. Each camera is analysed live, as the picture arrives: plates, "
               "people, vehicles and watch-list alerts show up a second or two after they happen, and everything "
               "found is searchable in the other tabs. The pictures of each moment with a detection are kept "
               f"{CFG['keep_days']:g} days (days with a watch-list match are kept); change this in Settings.")
    cams = store.cameras()
    running = live_running(store)
    b1, b2 = st.columns([1, 3])
    if running:
        b2.success("Live monitoring is running.")
        if b1.button("Stop live monitoring"):
            open(STOP_FILE, "w").close()
            st.info("Stopping…")
    else:
        b2.info("Live monitoring is stopped. Start it here. To start it by itself whenever this page opens, "
                "tick 'Start live monitoring when the page opens' in Settings.")
        if b1.button("Start live monitoring", type="primary", disabled=not cams):
            start_live()
            st.success("Starting… the cameras show 'recording' within a minute.")
    live_dvr_picker(cams)
    if cams:
        st.subheader("Cameras being watched")
    for c in cams:
        row = st.columns([2, 4, 1, 1])
        row[0].markdown(f"**{c['name']}**")
        row[1].caption(f"{c['status'] or 'not started'} · last seen {c['last_seen'] or '-'}")
        on = row[2].toggle("On", bool(c["enabled"]), key=f"cam_on_{c['id']}")
        if on != bool(c["enabled"]):
            store.update_camera(c["id"], enabled=int(on))
        if row[3].button("Remove", key=f"cam_rm_{c['id']}"):
            store.remove_camera(c["id"])
            st.rerun()
    if running and cams:
        live_view()
    with st.form("add_cam"):
        st.markdown("**Add a camera**")
        cn = st.text_input("Name", placeholder="Front gate")
        cu = st.text_input("Stream address", placeholder="rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101")
        st.caption("Hikvision: rtsp://USER:PASS@IP:554/Streaming/Channels/101 (channel 1 main stream, 201 = "
                   "channel 2) · Dahua: rtsp://USER:PASS@IP:554/cam/realmonitor?channel=1&subtype=0 · "
                   "Use the main stream for plates.")
        k1, k2 = st.columns(2)
        cp = k1.checkbox("Read plates", True, key="cam_p")
        cpp = k2.checkbox("Find people", True, key="cam_pp")
        t1, t2 = st.columns(2)
        test = t1.form_submit_button("Test connection")
        add = t2.form_submit_button("Add camera", type="primary")
        if (test or add) and cu:
            from targetcapture.live import test_camera

            with st.spinner("Connecting…"):
                ok, msg, jpg = test_camera(cu, timeout=30)
            if ok:
                st.image(jpg, caption="Picture from the camera", width=480)
            else:
                st.error(f"Could not connect: {msg}")
            if add and ok and cn:
                store.add_camera(cn, cu, cp, cpp)
                st.success(f"Added {cn}.")
                st.rerun()
            elif add and not cn:
                st.error("Give the camera a name.")

    with st.expander("Type a DVR / NVR address by hand (adds all its channels)",
                     expanded="dvr_msg" in st.session_state):
        if "dvr_msg" in st.session_state:
            st.success(st.session_state.pop("dvr_msg"))
            pic = st.session_state.pop("dvr_pic", None)
            how = st.session_state.pop("dvr_how", "")
            notes = st.session_state.pop("dvr_notes", [])
            if pic:
                st.image(pic, caption=f"Channel 1 ({how})", width=480)
                if how != "main stream":
                    st.info(f"The main stream sent no picture, so the channels use the {how}.")
            else:
                st.warning("The address and login are right, but channel 1 sent no picture (main stream, sub "
                           "stream and UDP were tried). Check that a camera is connected to channel 1, and in the "
                           "DVR's Video settings set the video type to H.264 instead of H.265 / H.265+.")
            if notes:
                with st.expander("Details (send these if you ask for help)"):
                    st.code("\n".join(notes))
        with st.form("add_dvr"):
            d1, d2, d3 = st.columns(3)
            dname = d1.text_input("Name", "DVR")
            dip = d2.text_input("IP address", placeholder="192.168.1.100")
            dbrand = d3.selectbox("Make", ["Hikvision / HiLook", "Dahua / Imou"])
            d4, d5, d6, d7 = st.columns(4)
            duser = d4.text_input("User name", "admin")
            dpass = d5.text_input("Password", type="password")
            dport = d6.number_input("RTSP port", 1, 65535, 554, help="554 unless it was changed in the DVR's network settings")
            dch = d7.number_input("Number of channels", 1, 64, 4)
            k1, k2 = st.columns(2)
            dp = k1.checkbox("Read plates", True, key="dvr_p")
            dpp = k2.checkbox("Find people", True, key="dvr_pp")
            if st.form_submit_button("Test channel 1 and add all channels", type="primary"):
                if not dip.strip() or not dpass:
                    st.error("Type the DVR's IP address and password.")
                else:
                    from targetcapture.live import (check_login, dvr_paths, hik_rtsp_port, rtsp_describe,
                                                    stream_url, sub_stream, test_camera)

                    from targetcapture.live import clean_host

                    ip, found, problem, log = clean_host(dip)[0], None, "", []
                    with st.status("Connecting to channel 1…", expanded=True) as box:
                        st.write("Checking the user name and password with the DVR's web login…")
                        login, info = check_login(dbrand, ip, duser, dpass)
                        port = int(dport)
                        if login is False:
                            problem = f"the DVR's web login says: {info}."
                        else:
                            if login:
                                st.write(f"Login is correct{' (' + info + ')' if info else ''}.")
                                set_port = hik_rtsp_port(ip, duser, dpass) if dbrand.startswith("Hik") else None
                                if set_port and set_port != port:
                                    st.write(f"The DVR's RTSP port is {set_port}, using that.")
                                    port = set_port
                            for label, path in dvr_paths(dbrand):
                                code, note = rtsp_describe(ip, port, path(1), duser, dpass)
                                log.append(f"{label} ({path(1)}): {code or 'no answer'} {note}".strip())
                                st.write(f"{label}: " + {200: "found", 401: "login refused", 404: "not on this DVR"}
                                         .get(code, f"no answer ({note})" if code is None else f"answer {code}"))
                                if code == 200:
                                    found = (label, path)
                                    break
                                if code is None:
                                    problem = (f"no answer on port {port}: check the IP address, that this PC is on "
                                               "the same network, and the RTSP port in the DVR's network settings.")
                                    break
                                if code == 401:
                                    problem = (note or "the DVR refuses this user name and password for video") + (
                                        ". The web login works, so check that this user may watch live video "
                                        "(User Management > Live View permission)." if login else ".")
                                    break  # same login on every path: do not lock the DVR with retries
                            if not found and not problem:
                                problem = "the DVR did not accept any of the usual stream addresses."
                        box.update(label="Connected" if found else "Could not connect",
                                   state="complete" if found else "error")
                    if not found:
                        st.error(f"Could not connect to channel 1: {problem}")
                        if log:
                            with st.expander("Details (send these if you ask for help)"):
                                st.code("\n".join(log))
                    else:
                        label, path = found

                        tries = [("main stream", path, "tcp", 20), ("main stream, read by Target Capture itself", path, "own", 20),
                                 ("sub stream", lambda ch, main=path: sub_stream(main(ch)), "tcp", 15),
                                 ("main stream over UDP", path, "udp", 15)]
                        ok, jpg, notes = False, None, []
                        with st.status("Getting a picture from channel 1…", expanded=True) as box:
                            for how, p, transport, secs in tries:
                                if how == "sub stream" and p(1) == path(1):
                                    continue
                                st.write(f"Trying the {how} (up to {secs} s)…")
                                ok, msg, jpg = test_camera(stream_url(ip, port, p(1), duser, dpass), secs, transport)
                                notes.append(f"{how} ({p(1)}): {msg}")
                                if ok:
                                    path = p
                                    break
                            box.update(label="Got a picture" if ok else "No picture from channel 1",
                                       state="complete" if ok else "error")

                        def make(ch):
                            return stream_url(ip, port, path(ch), duser, dpass)

                        st.session_state["dvr_pic"] = jpg if ok else None
                        st.session_state["dvr_notes"] = log + notes
                        st.session_state["dvr_how"] = how if ok else ""
                        have = {c["name"] for c in store.cameras()}
                        added = 0
                        for ch in range(1, int(dch) + 1):
                            name = f"{dname} ch{ch}"
                            if name not in have:
                                store.add_camera(name, make(ch), dp, dpp)
                                added += 1
                        st.session_state["dvr_msg"] = (
                            f"Added {added} channel(s) of {dname}. Switch off any channel you do not need; each "
                            "one uses part of the graphics card. Then click Start live monitoring.")
                        st.rerun()

with tab_faces:
    st.caption("Every face found in the indexed videos, grouped into people. Click a person to see "
               "every time they were seen.")
    c1, c2 = st.columns(2)
    f_thr = c1.slider("How alike faces must be to count as the same person", 0.35, 0.65, 0.45, 0.01,
                      help="Lower merges more (risk: two people become one). Higher splits more.")
    hide_oneoff = c2.checkbox("Hide people with only one face picture (often small or side-on faces)", True)
    db_mtime = os.path.getmtime(db_path) if os.path.exists(db_path) else 0
    everyone = cached_identities(db_path, db_mtime, f_thr)
    people = [p for p in everyone if not hide_oneoff or p["n_faces"] >= 2]
    sel = st.session_state.get("face_sel")
    if sel is not None:
        person = next((p for p in people if p["id"] == sel), None)
        if person:
            tracks = {tuple(k) if isinstance(k, list) else k: v for k, v in person["tracks"].items()}
            sightings = P.person_sightings(store, {**person, "tracks": tracks}, threshold=f_thr)
            out = os.path.join(RESULTS, f"face_person_{sel}")
            rows = person_report(store, [f"person #{sel}"], sightings,
                                 {"faces_indexed": person["n_faces"]}, f_thr, out)
            hdr = st.columns([4, 1])
            hdr[0].subheader(f"Person #{sel}: seen {len(rows)} time(s), {person['n_faces']} face detections")
            if hdr[1].button("Close", key="face_close"):
                st.session_state.face_sel = None
                st.rerun()
            st.caption(f"Report: {os.path.join(out, 'report.html')}")
            downloads(rows, out, f"Person #{sel}", f"person_{sel}")
            show_rows(rows, out, ["face_image"])
            st.divider()
    st.subheader(f"{len(people)} people")
    if len(everyone) > len(people):
        st.caption(f"{len(everyone) - len(people)} more seen in only one face picture: untick 'Hide people "
                   "with only one face picture' to see them.")
    cols = st.columns(6)
    for i, p in enumerate(people[pager(len(people), "face_page")]):
        b = p["best"]
        media = store.media(b["media_id"])
        with cols[i % 6]:
            img = thumbnail(b, store, THUMBS, box=("fx1", "fy1", "fx2", "fy2"), pad=0.35, min_w=140)
            if img:
                st.image(img, width="stretch")
            tracks = {tuple(k) if isinstance(k, list) else k: v for k, v in p["tracks"].items()}
            seen = len(P.person_sightings(store, {**p, "tracks": tracks}, threshold=f_thr))
            st.caption(f"#{p['id']} · seen {seen}x · first at {fmt_offset(p['first'])}")
            if st.button(f"Show #{p['id']}", key=f"face_{p['id']}"):
                st.session_state.face_sel = p["id"]
                st.rerun()

with tab_all:
    st.subheader("Read plates")
    min_r = st.number_input("Hide plates read fewer times than", 1, 10, 2)
    vfilter = st.text_input("Filter by vehicle (e.g. corolla, white toyota, Erbil)").lower().split()
    sightings = [s for s in cached_plate_sightings(db_path, plate_data_sig(store)) if s["reads"] >= min_r and all(
        w in " ".join(str(s[k] or "") for k in ("vehicle_model", "vehicle_color", "vehicle_type", "region")).lower()
        for w in vfilter)]
    st.caption("Tick **View** to open a plate with its best capture and a second one taken after it.")
    pick = plate_table(sightings, "plates_table")
    if pick:
        view_plate(pick, "plate_view")
    if sightings and st.button(f"Prepare Excel / PDF of all {len(sightings)} plates"):
        out = os.path.join(RESULTS, "all_plates")
        shutil.rmtree(out, ignore_errors=True)
        with st.spinner("Saving pictures…"):
            rows = plate_report(store, "all plates", sightings, out)
        downloads(rows, out, "All plates", "all_plates")

    unread = cached_unread_captures(db_path, plate_data_sig(store))
    st.subheader(f"Plates that could not be read ({len(unread)})")
    st.caption("Plates the system found but could not read (other alphabets, blur, too small). "
               "Click one to see the whole frame.")
    sel = st.session_state.get("cap_sel")
    if sel is not None and sel < len(unread):
        c = unread[sel]
        media = store.media(c["media_id"])
        img = thumbnail(c["best"], store, THUMBS, full=True)
        hdr = st.columns([4, 1])
        hdr[0].markdown(f"**Capture {sel + 1}** · {fmt_offset(c['start'])}"
                        f"{' (' + clock_time(media, c['start']) + ')' if clock_time(media, c['start']) else ''}"
                        f" · {os.path.basename(media['path'])}")
        if hdr[1].button("Close", key="cap_close"):
            st.session_state.cap_sel = None
            st.rerun()
        if img:
            st.image(img)
    cols = st.columns(6)
    sl = pager(len(unread), "cap_page")
    for i, c in enumerate(unread[sl]):
        n = (sl.start or 0) + i
        media = store.media(c["media_id"])
        with cols[i % 6]:
            img = thumbnail(c["best"], store, THUMBS, pad=0.1, min_w=200)
            if img:
                st.image(img, width="stretch")
            vehicle = f"{c['vehicle_color'] or ''} {c['vehicle_model'] or c['vehicle_type'] or ''}".strip()
            st.caption(f"{fmt_offset(c['start'])} · {vehicle or 'vehicle ?'}"
                       f"{' · seen ' + str(c['detections']) + 'x' if c['detections'] > 1 else ''}")
            if st.button("Open", key=f"cap_{n}"):
                st.session_state.cap_sel = n
                st.rerun()

with tab_teach:
    st.caption("Help the system: type the correct text for plates it could not read or read wrongly. "
               "Your answers are searchable at once, and the system uses them to recognise the same plate "
               "(and the same reading mistake) in the future.")
    mode = st.radio("What to check", ["unread", "check", "find"], horizontal=True, key="teach_mode",
                    format_func=lambda m: {"unread": "Plates it could not read",
                                           "check": "Check read plates (least sure first)",
                                           "find": "Correct a specific plate"}[m])
    sig = plate_data_sig(store)
    if mode == "unread":
        queue = list(cached_unread_captures(db_path, sig))
    else:
        queue = [s for s in cached_plate_sightings(db_path, sig) if s["source"] != "taught"]
        if mode == "find":
            q = st.text_input("Plate as the system shows it (from the All plates table)", key="teach_find")
            queue = [s for s in queue if q and L.match_score(q, s["text"], 1)[0]]
        else:
            queue.sort(key=lambda s: (s["conf"], s["reads"]))

    last = st.session_state.get("teach_last")
    if last:  # follow-ups to the correction just saved
        with st.container(border=True):
            st.success(f"Saved **{last['label'] or 'not a plate'}**.")
            if last["same_mistake"]:
                n = len(last["same_mistake"])
                if st.button(f"Also fix {n} other read(s) that were read as \"{last['ocr_text']}\""):
                    store.set_plate_text(last["same_mistake"], last["label"], "taught")
                    last["same_mistake"] = []
                    st.rerun()
            if last["sig"] is not None and last["label"]:
                pool = list(cached_unread_captures(db_path, sig)) + [
                    s for s in cached_plate_sightings(db_path, sig) if s["source"] is None and s["conf"] < 0.9]
                sims = T.similar_captures(store, last["sig"], pool)[:12]
                if sims:
                    st.markdown(f"These look like the same plate. Tick the ones that are **{last['label']}**:")
                    cols = st.columns(6)
                    picked = []
                    for i, (sim, c) in enumerate(sims):
                        media = store.media(c["media_id"])
                        with cols[i % 6]:
                            img = thumbnail(c["best"], store, THUMBS, pad=0.1, min_w=200)
                            if img:
                                st.image(img, width="stretch")
                            if st.checkbox(f"{fmt_offset(c['start'])} · {sim:.0%} alike", key=f"sim_{i}_{c['ids'][:1]}"):
                                picked.append(c)
                    if picked and st.button(f"Apply {last['label']} to {len(picked)} selected"):
                        for c in picked:
                            store.set_plate_text(c["ids"], last["label"], "taught")
                        st.session_state.teach_last = None
                        st.rerun()
            if st.button("Done", key="teach_done"):
                st.session_state.teach_last = None
                st.rerun()

    if not queue:
        st.info("Nothing to check here." if mode != "find" else "Type a plate above to find it.")
    else:
        i = min(st.session_state.get(f"teach_i_{mode}", 0), len(queue) - 1)
        c = queue[i]
        media = store.media(c["media_id"])
        clock = clock_time(media, c["start"])
        st.markdown(f"**{i + 1} of {len(queue)}** · {fmt_offset(c['start'])}{' (' + clock + ')' if clock else ''}"
                    f" · {os.path.basename(media['path'])}")
        left, right = st.columns([1, 2])
        crop = thumbnail(c["best"], store, THUMBS, pad=0.1, min_w=360)
        frame = thumbnail(c["best"], store, THUMBS, full=True)
        if crop:
            left.image(crop, width="stretch")
        if frame:
            right.image(frame, width="stretch")
        else:
            right.info("This frame could not be found in the video again. Re-index the video to fix this.")
        current = c.get("text") or ""
        guess = current or c.get("ocr_guess") or ""
        vehicle = f"{c['vehicle_color'] or ''} {c['vehicle_model'] or c['vehicle_type'] or ''}".strip()
        left.caption(f"System read: **{current or 'nothing'}**"
                     f"{' · guess: ' + c['ocr_guess'] if not current and c.get('ocr_guess') else ''}"
                     f"{' · ' + vehicle if vehicle else ''}")
        with left.form(f"teach_form_{mode}_{i}"):
            label = st.text_input("Correct plate text (Arabic or English digits both fine)", value=guess)
            b1, b2, b3 = st.columns(3)
            save = b1.form_submit_button("Save", type="primary")
            notplate = b2.form_submit_button("No plate")
            skip = b3.form_submit_button("Skip")
        if save or notplate:
            if save and not L.normalize(label):
                st.warning("Type the plate text first, or click No plate.")
            else:
                st.session_state.teach_last = T.teach(store, c["ids"], c["best"], label, not_a_plate=bool(notplate))
                st.rerun()
        if skip:
            st.session_state[f"teach_i_{mode}"] = (i + 1) % len(queue)
            st.rerun()
    with st.expander("Export what you taught (for training a better reader)"):
        n_labels = len([lab for lab in store.labels() if lab["label"]])
        st.caption(f"{n_labels} plate(s) taught so far.")
        if st.button("Export to results/taught_plates", disabled=not n_labels):
            n, missing = T.export(store, os.path.join(RESULTS, "taught_plates"))
            st.success(f"Wrote {n} plate pictures and labels.csv to {os.path.join(RESULTS, 'taught_plates')}")
            if missing:
                st.info(f"{missing} taught plate(s) were skipped: their videos were removed from the index "
                        "before this version, so their pictures are gone. They still correct new readings.")

with tab_tips:
    st.markdown(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "TIPS.md"), encoding="utf-8").read())

def start_update():
    """Run the updater in its own window from a copy (the update replaces update_windows.bat),
    then close this web page so its files can be replaced; the updater starts it again."""
    copy = os.path.join(tempfile.gettempdir(), "target_capture_update.bat")
    shutil.copyfile(os.path.join(APP_DIR, "update_windows.bat"), copy)
    subprocess.Popen([copy, "restart", APP_DIR], cwd=APP_DIR, creationflags=subprocess.CREATE_NEW_CONSOLE)
    import threading

    threading.Timer(2.0, os._exit, [0]).start()


STARTUP_FILE = os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs",
                            "Startup", "Target Capture.cmd")


def set_windows_startup(on: bool):
    """Start the web page (minimized) when Windows starts, with a small file in the Startup folder."""
    if on:
        with open(STARTUP_FILE, "w", newline="\r\n") as f:
            f.write(f'@echo off\ncd /d "{APP_DIR}"\nstart "Target Capture" /min run_windows.bat\n')
    elif os.path.exists(STARTUP_FILE):
        os.remove(STARTUP_FILE)


def run_and_restart(bat: str):
    """Run one of the .bat files in its own window from a temp copy with "restart <app folder>": it
    stops this page, does its work and starts the page again."""
    copy = os.path.join(tempfile.gettempdir(), f"target_capture_{bat}")
    shutil.copyfile(os.path.join(APP_DIR, bat), copy)
    subprocess.Popen([copy, "restart", APP_DIR], cwd=APP_DIR, creationflags=subprocess.CREATE_NEW_CONSOLE)
    import threading

    threading.Timer(2.0, os._exit, [0]).start()


def restart_page():
    """Close this web page and start it again (applies the largest-upload setting)."""
    copy = os.path.join(tempfile.gettempdir(), "target_capture_restart.cmd")
    with open(copy, "w", newline="\r\n") as f:
        f.write(f'@echo off\ntimeout /t 3 /nobreak >nul\ncd /d "{APP_DIR}"\nstart "Target Capture" run_windows.bat\n')
    subprocess.Popen(["cmd", "/c", copy], cwd=APP_DIR, creationflags=subprocess.CREATE_NO_WINDOW)
    import threading

    threading.Timer(2.0, os._exit, [0]).start()


with tab_settings:
    v = app_version()
    st.subheader("Version")
    st.markdown(f"**Target Capture {v['version']}**" +
                (f"  \nInstalled update: {v['date']} ({v['commit']}) · {v['subject']}" if v["commit"] else
                 "  \nThis folder is not linked to GitHub yet (run update_windows.bat once)."))
    if st.button("Check for updates", icon=":material/sync:"):
        with st.spinner("Asking GitHub…"):
            st.session_state["update_check"] = C.check_update()
    chk = st.session_state.get("update_check")
    if chk:
        if not chk["ok"]:
            st.error(chk["error"])
        elif not chk["changes"]:
            st.success("You have the latest version.")
        else:
            st.info(f"{len(chk['changes'])} update(s) available:")
            st.markdown("\n".join(f"* {d}: {msg}" for d, msg in chk["changes"]))
            if os.name == "nt":
                live_on = live_running(store)
                if live_on:
                    st.warning("Stop live monitoring (Live cameras tab) before updating.")
                if st.button("Update now", type="primary", disabled=live_on):
                    st.warning("Updating in a new window. This page closes now and opens again when the "
                               "update is done (a minute or two; longer when new libraries are needed).")
                    start_update()
            else:
                st.code("git pull && pip install -r requirements.txt", language="bash")

    st.subheader("Use from other devices")
    import socket

    port = st.get_option("server.port")
    addrs = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("10.255.255.255", 1))
            addrs.add(sk.getsockname()[0])
    except OSError:
        pass
    try:
        addrs |= {a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    except OSError:
        pass
    addrs = sorted(a for a in addrs if not a.startswith("127."))
    tail = [a for a in addrs if a.startswith("100.")]
    lan = [a for a in addrs if not a.startswith("100.")]
    st.markdown(
        "The work (downloads, indexing, live cameras) always runs on this PC with its graphics card; other "
        "devices only show the page.\n\n"
        + (f"* **Same network** (office Wi-Fi): " + ", ".join(f"`http://{a}:{port}`" for a in lan) + "\n" if lan else "")
        + ("* **From anywhere (Tailscale):** " + ", ".join(f"`http://{a}:{port}`" for a in tail) + "\n" if tail else
           "* **From anywhere:** install [Tailscale](https://tailscale.com/download) (free) on this PC and on your "
           "phone or laptop, signed in with the same account. Its address for this PC (100.x.x.x) then shows here. "
           "It is private: only your own devices can open the page.\n"))
    st.caption(f"Do not open port {port} on the router to the whole internet: the page shows faces and plates.")
    p1, p2, p3 = st.columns([2, 1, 1])
    new_pw = p1.text_input("Page password for other devices" + (" (set)" if CFG["page_password"] else ""),
                           type="password", help="Asked when the page is opened from another device, not on this PC.")
    if p2.button("Set password", help="At least 6 characters."):
        if len(new_pw) < 6:
            st.error("Use at least 6 characters.")
        else:
            C.save({"page_password": C.hash_password(new_pw)})
            st.success("Password set. Other devices are asked for it from now on.")
    if CFG["page_password"] and p3.button("Remove password", disabled=not from_this_pc(),
                                          help="Only from this PC."):
        C.save({"page_password": ""})
        st.success("Password removed.")
    if not CFG["page_password"]:
        st.warning("No page password yet: set one before using the page from other devices.")

    if os.name == "nt":
        st.subheader("Start and graphics card")
        st.caption("Everything the .bat files in the Target Capture folder do, from here. You only need "
                   "run_windows.bat (or the Start with Windows option) to open this page.")
        on = st.toggle("Open Target Capture when Windows starts", os.path.exists(STARTUP_FILE),
                       help="Opens this page (its window minimized) after you sign in to Windows. Together with "
                            "'Start live monitoring when the page opens' below, live cameras start by themselves "
                            "after a restart or power cut.")
        if on != os.path.exists(STARTUP_FILE):
            set_windows_startup(on)
            st.rerun()
        from targetcapture.gpu import describe

        gpu_on = os.path.exists(os.path.join(APP_DIR, ".venv", "gpu.txt"))
        st.markdown(f"Graphics card: **{describe()}**" + (" · GPU support installed." if gpu_on else ""))
        g1, g2, g3 = st.columns(3)
        live_on = live_running(store)
        if g1.button("Reinstall graphics-card support" if gpu_on else "Install graphics-card support (NVIDIA)",
                     disabled=live_on, help="Same as setup_gpu_windows.bat: downloads about 3 GB, then opens "
                                            "this page again. Needs the NVIDIA driver."):
            st.warning("Installing in a new window. This page closes now and opens again when it is done.")
            run_and_restart("setup_gpu_windows.bat")
        if g2.button("Restart this page", help="Closes and opens the page again (needed after changing the "
                                                "largest upload). Live monitoring keeps running."):
            st.warning("Restarting… the page opens again in a few seconds.")
            restart_page()
        if live_on:
            g3.caption("Stop live monitoring before installing graphics-card support.")

    st.subheader("Defaults")
    st.caption("Saved in settings.json. Updates keep it. Used by new indexing runs, searches and live cameras.")
    with st.form("settings"):
        st.markdown("**Indexing**")
        c1, c2, c3 = st.columns(3)
        n = {}
        n["read_plates"] = c1.checkbox("Read plates", CFG["read_plates"])
        n["find_people"] = c2.checkbox("Find people", CFG["find_people"])
        n["car_models"] = c3.checkbox("Car make and model", CFG["car_models"])
        speeds = list(C.SPEED_NAMES)
        n["speed"] = c1.radio("Indexing speed", speeds, index=speeds.index(CFG["speed"]), format_func=C.SPEED_NAMES.get,
                              help="Fast looks at fewer frames per second (plates 3, people 2): most plates and "
                                   "faces are still seen several times. Custom uses the two frame rates below.")
        n["plate_fps"] = c2.slider("Custom: frames per second for plates", 1.0, 10.0, float(CFG["plate_fps"]),
                                   help="Higher reads more plates on fast cars but is slower.")
        n["person_fps"] = c3.slider("Custom: frames per second for people", 1.0, 10.0, float(CFG["person_fps"]))
        n["min_ocr_conf"] = c3.slider("Minimum plate read confidence", 0.2, 0.9, float(CFG["min_ocr_conf"]), 0.05,
                                      help="Lower keeps more (and more wrong) reads.")
        n["plate_script"] = c1.radio("Plate script", SCRIPTS, index=SCRIPTS.index(CFG["plate_script"]),
                                     format_func={"auto": "Auto (English and Arabic)", "latin": "English only",
                                                  "arabic": "Arabic / Kurdish only"}.get)
        sizes = ["auto", "fast", "accurate"]
        n["car_model_size"] = c3.radio("Car model recognizer", sizes, index=sizes.index(CFG["car_model_size"]),
                                       format_func={"auto": "Auto (accurate on a GPU with 6 GB or more)",
                                                    "fast": "Fast (600 MB)",
                                                    "accurate": "Accurate, slower (1.7 GB download)"}.get)
        n["vehicle_crops"] = c2.checkbox("Read plates inside each vehicle (slower)", CFG["vehicle_crops"])
        n["skip_empty_road"] = c2.checkbox("Skip plate reading when no vehicle is in view", CFG["skip_empty_road"])
        n["keep_frames"] = c1.checkbox("Keep pictures of plates and faces while indexing", CFG["keep_frames"],
                                       help="Results then show at once and on the exact frame. Uses disk "
                                            "space (about 100-300 KB per picture); see Storage below.")
        n["gpu_decode"] = c2.checkbox("Decode video on the graphics card", CFG["gpu_decode"],
                                      help="Less work for the processor on long 1080p/4K videos. Falls back "
                                           "to normal decoding by itself when the card or driver cannot.")
        st.markdown("**Searching and watch list**")
        c1, c2, c3, c4 = st.columns(4)
        n["plate_max_dist"] = c1.number_input("Plate search: allowed differences", 0, 3, int(CFG["plate_max_dist"]))
        n["watch_plate_dist"] = c2.number_input("Watch list: plate differences", 0, 3, int(CFG["watch_plate_dist"]))
        n["face_threshold"] = c3.slider("Watch list: face match strictness", 0.3, 0.7, float(CFG["face_threshold"]),
                                        0.01, help="Higher means fewer, surer matches.")
        n["page_size"] = c4.number_input("Pictures per page", 12, 120, int(CFG["page_size"]), 12)
        st.markdown("**Live cameras**")
        c1, c2 = st.columns(2)
        n["keep_days"] = c1.number_input("Keep recordings (days)", 1.0, 365.0, float(CFG["keep_days"]), 1.0)
        n["live_autostart"] = c1.checkbox("Start live monitoring when the page opens", CFG["live_autostart"],
                                          help="No need to press Start (or run run_live_windows.bat) after "
                                               "opening the page or restarting the computer.")
        n["live_record"] = c2.checkbox("Also record the cameras (for watching later; the live analysis does "
                                       "not need it)", CFG["live_record"])
        n["segment_seconds"] = c2.number_input("Recorded piece length (seconds)", 30, 600,
                                               int(CFG["segment_seconds"]), 30)
        st.markdown("**Web page**")
        c1, c2 = st.columns(2)
        n["db_path"] = c1.text_input("Index database file", CFG["db_path"])
        n["max_upload_mb"] = c2.number_input("Largest upload (MB)", 200, 100000, int(CFG["max_upload_mb"]), 100,
                                             help="Takes effect after restarting the page (Restart this page, above).")
        b1, b2 = st.columns([1, 5])
        save = b1.form_submit_button("Save", type="primary")
        reset = b2.form_submit_button("Back to defaults")
    if save or reset:
        C.save({**C.DEFAULTS, "page_password": CFG["page_password"]} if reset else n)
        st.toast("Settings saved.")
        st.rerun()

    st.subheader("This computer")
    import platform

    st.markdown(f"* {device_line()}\n* Python {platform.python_version()} on {platform.system()} {platform.release()}")
    libs = []
    for mod in ("torch", "onnxruntime", "ultralytics", "fast_alpr", "insightface", "easyocr", "streamlit"):
        try:
            libs.append(f"{mod} {__import__(mod).__version__}")
        except Exception:
            pass
    st.caption(" · ".join(libs))
    st.subheader("Storage")
    db_dir = os.path.dirname(os.path.abspath(db_path))
    places = {"Index database": db_path, "Uploaded videos": UPLOADS, "Repaired DVR videos": os.path.join(db_dir, "fixed_videos"),
              "Kept plate and face pictures": os.path.join(db_dir, "kept_frames"),
              "DVR downloads": CFG["dvr_folder"] or os.path.join(db_dir, "dvr_videos"),
              "Live recordings": "recordings", "Live camera pictures": os.path.join(db_dir, "live_frames"),
              "Results and pictures": RESULTS}
    sizes = {k: folder_size(p) for k, p in places.items() if os.path.exists(p)}
    free = shutil.disk_usage(APP_DIR).free
    st.markdown("\n".join(f"* {k}: **{human(b)}**" for k, b in sizes.items()) + f"\n* Free disk space: **{human(free)}**")
    counts = store.db.execute("SELECT (SELECT COUNT(*) FROM media), (SELECT COUNT(*) FROM plate_reads), "
                              "(SELECT COUNT(*) FROM person_dets)").fetchone()
    st.caption(f"{counts[0]} videos/images indexed · {counts[1]} plate reads · {counts[2]} person detections")
    k1, k2, k3 = st.columns(3)
    if k1.button("Clear saved result pictures", help="Search results and reports in the results folder. "
                 "They are made again when you search; nothing in the index is lost."):
        shutil.rmtree(RESULTS, ignore_errors=True)
        st.success("Cleared.")
    live_on = live_running(store)

    def clear_folder(folder):
        """Remove every indexed video / camera day kept in `folder`, then the folder's files."""
        root = os.path.abspath(folder) + os.sep
        for m in store.db.execute("SELECT id, path FROM media").fetchall():
            if m["path"] and os.path.abspath(m["path"]).startswith(root):
                remove_media(m["id"])
        shutil.rmtree(folder, ignore_errors=True)
        left = folder_size(folder) if os.path.exists(folder) else 0
        return left

    with k2.popover("Delete live recordings", disabled=live_on or not os.path.exists("recordings"),
                    help="Stop live monitoring first." if live_on else "Recorded camera video (the live analysis "
                         "does not need it)."):
        st.write(f"Delete all recorded camera video ({human(sizes.get('Live recordings', 0))})? "
                 "Anything indexed from it is removed from the index too.")
        if st.button("Yes, delete", key="del_rec", type="primary"):
            left = clear_folder("recordings")
            st.session_state["storage_msg"] = "Deleted." if not left else f"Deleted, except {human(left)} of files in use."
            st.rerun()
    frames = places["Live camera pictures"]
    with k3.popover("Delete live camera results", disabled=live_on or not os.path.exists(frames),
                    help="Stop live monitoring first." if live_on else "Plates, people and pictures found by the "
                         "live cameras."):
        st.write(f"Delete everything the live cameras found, with its pictures "
                 f"({human(sizes.get('Live camera pictures', 0))})? The cameras themselves stay set up.")
        if st.button("Yes, delete", key="del_frames", type="primary"):
            left = clear_folder(frames)
            st.session_state["storage_msg"] = "Deleted." if not left else f"Deleted, except {human(left)} of files in use."
            st.rerun()
    delete_downloads_button("del_dl_storage")
    if live_on:
        st.caption("Stop live monitoring (Live cameras tab) to delete live recordings or results.")
    if "storage_msg" in st.session_state:
        st.success(st.session_state.pop("storage_msg"))
