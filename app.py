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
st.title("Target Capture")
db_path = st.sidebar.text_input("Index database", CFG["db_path"])
store = Store(db_path)

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

(tab_index, tab_plate, tab_person, tab_car, tab_watch, tab_cams, tab_faces, tab_all, tab_teach,
 tab_tips, tab_settings) = st.tabs(["1 · Upload & index", "2 · Find a plate", "3 · Find a person", "4 · Find a car",
                      "Watch list", "Live cameras", "All faces", "All plates", "Teach", "Best results", "Settings"])

with tab_index:
    up = st.file_uploader("Video or image", type=["mp4", "avi", "mkv", "mov", "dav", "h264", "ts",
                                                  "jpg", "jpeg", "png", "bmp", "webp"])
    path_in = st.text_input("…or a path to a file already on this machine (best for multi-hour DVR files)")
    c1, c2, c3 = st.columns(3)
    do_plates = c1.checkbox("Read plates", CFG["read_plates"])
    do_persons = c2.checkbox("Find people", CFG["find_people"])
    clock = c3.text_input("DVR clock at first frame (optional)", placeholder="2026-09-30 08:00:00")
    with st.expander("Advanced"):
        plate_fps = st.slider("Frames per second analysed for plates", 1.0, 10.0, float(CFG["plate_fps"]))
        person_fps = st.slider("Frames per second analysed for people", 1.0, 10.0, float(CFG["person_fps"]))
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
        bars = {}

        def progress(kind, i, total, found, t):
            if kind not in bars:
                bars[kind] = st.progress(0.0)
            bars[kind].progress(min(1.0, i / max(1, total)),
                                text=f"{kind}: {fmt_offset(t)} analysed, {found} detections")

        res = index_media(store, path, plates=do_plates, persons=do_persons, plate_fps=plate_fps,
                          person_fps=person_fps, clock_start=clock or None, vehicle_crops=vehicle_crops, plate_script=plate_script, car_models=car_models,
                          reindex=reindex, progress=progress, skip_empty_road=skip_empty,
                          min_ocr_conf=CFG["min_ocr_conf"])
        st.success(f"Indexed {os.path.basename(path)}. Now search it in the next tabs.")
        if res.get("alerts"):
            st.error(f"{len(res['alerts'])} watch-list match(es) in this video: see the Watch list tab.")

with tab_plate:
    q = st.text_input("Plate number (wildcards allowed: 12?BC* )")
    c1, c2 = st.columns(2)
    max_dist = c1.number_input("Allowed character differences", 0, 3, int(CFG["plate_max_dist"]))
    min_reads = c2.number_input("Minimum reads per sighting", 1, 10, 1)
    if st.button("Search plate", disabled=not q):
        hits = L.search(store, q, max_dist=max_dist, min_reads=min_reads)
        out = os.path.join(RESULTS, "plate_" + "".join(ch for ch in q if ch.isalnum()))
        rows = plate_report(store, q, hits, out)
        st.subheader(f"{len(rows)} sighting(s)")
        st.caption(f"Full report saved to {os.path.join(out, 'report.html')}")
        downloads(rows, out, f"Plate search {q}", "plate_search")
        show_rows(rows, out, ["plate_image", "second_plate_image"])

with tab_person:
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

with tab_car:
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

with tab_cams:
    st.caption("Connect DVR / NVR / IP cameras. Each camera is recorded in 1-minute pieces and indexed "
               "straight away: plates, people, vehicles and watch-list alerts about a minute after "
               f"they happen. Recordings are kept {CFG['keep_days']:g} days (pieces with a watch-list match are "
               "kept); change this in Settings.")
    stop_file = os.path.join(os.path.dirname(os.path.abspath(db_path)), "live.stop")
    cams = store.cameras()
    now = dt.datetime.now()
    running = any(c["last_seen"] and (now - dt.datetime.fromisoformat(c["last_seen"])).total_seconds() < 90
                  for c in cams)
    b1, b2 = st.columns([1, 3])
    if running:
        b2.success("Live monitoring is running.")
        if b1.button("Stop live monitoring"):
            open(stop_file, "w").close()
            st.info("Stopping after the minute it is working on…")
    else:
        b2.info("Live monitoring is stopped. Start it here, or double-click run_live_windows.bat.")
        if b1.button("Start live monitoring", type="primary", disabled=not cams):
            if os.path.exists(stop_file):
                os.remove(stop_file)
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            subprocess.Popen([sys.executable, "-m", "targetcapture", "--db", db_path, "live",
                              "--stop-file", stop_file], creationflags=flags,
                             stdout=open("live.log", "a"), stderr=subprocess.STDOUT)
            st.success("Starting… the cameras show 'recording' within a minute.")
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
                ok, msg, jpg = test_camera(cu)
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

with tab_faces:
    st.caption("Every face found in the indexed videos, grouped into people. Click a person to see "
               "every time they were seen.")
    c1, c2 = st.columns(2)
    f_thr = c1.slider("How alike faces must be to count as the same person", 0.35, 0.65, 0.45, 0.01,
                      help="Lower merges more (risk: two people become one). Higher splits more.")
    hide_oneoff = c2.checkbox("Hide people only caught in single frames (small or side-on faces)", True)
    db_mtime = os.path.getmtime(db_path) if os.path.exists(db_path) else 0
    people = [p for p in cached_identities(db_path, db_mtime, f_thr)
              if not hide_oneoff or p["max_track_faces"] >= 3]
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
    sightings = [s for s in L.group_sightings(store.plate_reads()) if s["reads"] >= min_r and all(
        w in " ".join(str(s[k] or "") for k in ("vehicle_model", "vehicle_color", "vehicle_type", "region")).lower()
        for w in vfilter)]
    table = [{
        "plate": s["text"], "region / plate words": s["region"] or "",
        "start": fmt_offset(s["start"]), "end": fmt_offset(s["end"]),
        "clock": clock_time(store.media(s["media_id"]), s["start"]) or "", "reads": s["reads"],
        "conf": s["conf"], "vehicle": f"{s['vehicle_color'] or ''} {s['vehicle_model'] or s['vehicle_type'] or ''}".strip(),
        "source": {"taught": "typed by you", "learned": "learned"}.get(s["source"], "read"),
        "video": os.path.basename(store.media(s["media_id"])["path"]),
        "View": False,
    } for s in sightings]
    st.caption("Tick **View** to open a plate with its best capture and a second one taken after it.")
    edited = st.data_editor(table, width="stretch", hide_index=True, key=f"plates_table_{len(table)}",
                            disabled=[k for k in (table[0] if table else {}) if k != "View"],
                            column_config={"View": st.column_config.CheckboxColumn("View", width="small")})
    picked = [i for i, row in enumerate(edited) if row["View"]]
    if picked:
        s = sightings[picked[-1]]
        out = os.path.join(RESULTS, "plate_view_" + "".join(ch for ch in s["text"] if ch.isalnum()))
        rows = plate_report(store, s["text"], [s], out)
        st.markdown(f"**{s['text']}** · {fmt_offset(s['start'])} to {fmt_offset(s['end'])}")
        downloads(rows, out, f"Plate {s['text']}", "plate_view")
        show_rows(rows, out, ["plate_image", "second_plate_image"])
    if sightings and st.button(f"Prepare Excel / PDF of all {len(sightings)} plates"):
        out = os.path.join(RESULTS, "all_plates")
        shutil.rmtree(out, ignore_errors=True)
        with st.spinner("Saving pictures…"):
            rows = plate_report(store, "all plates", sightings, out)
        downloads(rows, out, "All plates", "all_plates")

    unread = L.group_unread(store.plate_reads(unread=True))
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
    if mode == "unread":
        queue = L.group_unread(store.plate_reads(unread=True))
    else:
        queue = [s for s in L.group_sightings(store.plate_reads()) if s["source"] != "taught"]
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
                pool = L.group_unread(store.plate_reads(unread=True)) + [
                    s for s in L.group_sightings(store.plate_reads()) if s["source"] is None and s["conf"] < 0.9]
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
            n = T.export(store, os.path.join(RESULTS, "taught_plates"))
            st.success(f"Wrote {n} plate pictures and labels.csv to {os.path.join(RESULTS, 'taught_plates')}")

with tab_tips:
    st.markdown(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "TIPS.md"), encoding="utf-8").read())

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


def start_update():
    """Run the updater in its own window from a copy (the update replaces update_windows.bat),
    then close this web page so its files can be replaced; the updater starts it again."""
    copy = os.path.join(tempfile.gettempdir(), "target_capture_update.bat")
    shutil.copyfile(os.path.join(APP_DIR, "update_windows.bat"), copy)
    subprocess.Popen([copy, "restart", APP_DIR], cwd=APP_DIR, creationflags=subprocess.CREATE_NEW_CONSOLE)
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
                live_on = any(c["last_seen"] and (dt.datetime.now() - dt.datetime.fromisoformat(c["last_seen"]))
                              .total_seconds() < 90 for c in store.cameras())
                if live_on:
                    st.warning("Stop live monitoring (Live cameras tab) before updating.")
                if st.button("Update now", type="primary", disabled=live_on):
                    st.warning("Updating in a new window. This page closes now and opens again when the "
                               "update is done (a minute or two; longer when new libraries are needed).")
                    start_update()
            else:
                st.code("git pull && pip install -r requirements.txt", language="bash")

    st.subheader("Defaults")
    st.caption("Saved in settings.json. Updates keep it. Used by new indexing runs, searches and live cameras.")
    with st.form("settings"):
        st.markdown("**Indexing**")
        c1, c2, c3 = st.columns(3)
        n = {}
        n["read_plates"] = c1.checkbox("Read plates", CFG["read_plates"])
        n["find_people"] = c2.checkbox("Find people", CFG["find_people"])
        n["car_models"] = c3.checkbox("Car make and model", CFG["car_models"])
        n["plate_fps"] = c1.slider("Frames per second for plates", 1.0, 10.0, float(CFG["plate_fps"]),
                                   help="Higher reads more plates on fast cars but is slower.")
        n["person_fps"] = c2.slider("Frames per second for people", 1.0, 10.0, float(CFG["person_fps"]))
        n["min_ocr_conf"] = c3.slider("Minimum plate read confidence", 0.2, 0.9, float(CFG["min_ocr_conf"]), 0.05,
                                      help="Lower keeps more (and more wrong) reads.")
        n["plate_script"] = c1.radio("Plate script", SCRIPTS, index=SCRIPTS.index(CFG["plate_script"]),
                                     format_func={"auto": "Auto (English and Arabic)", "latin": "English only",
                                                  "arabic": "Arabic / Kurdish only"}.get)
        n["vehicle_crops"] = c2.checkbox("Read plates inside each vehicle (slower)", CFG["vehicle_crops"])
        n["skip_empty_road"] = c2.checkbox("Skip plate reading when no vehicle is in view", CFG["skip_empty_road"])
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
        n["segment_seconds"] = c2.number_input("Recorded piece length (seconds)", 30, 600,
                                               int(CFG["segment_seconds"]), 30)
        st.markdown("**Web page**")
        c1, c2 = st.columns(2)
        n["db_path"] = c1.text_input("Index database file", CFG["db_path"])
        n["max_upload_mb"] = c2.number_input("Largest upload (MB)", 200, 100000, int(CFG["max_upload_mb"]), 100,
                                             help="Takes effect after closing and starting the page again.")
        b1, b2 = st.columns([1, 5])
        save = b1.form_submit_button("Save", type="primary")
        reset = b2.form_submit_button("Back to defaults")
    if save or reset:
        C.save(C.DEFAULTS if reset else n)
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
              "Live recordings": "recordings", "Results and pictures": RESULTS}
    sizes = {k: folder_size(p) for k, p in places.items() if os.path.exists(p)}
    free = shutil.disk_usage(APP_DIR).free
    st.markdown("\n".join(f"* {k}: **{human(b)}**" for k, b in sizes.items()) + f"\n* Free disk space: **{human(free)}**")
    counts = store.db.execute("SELECT (SELECT COUNT(*) FROM media), (SELECT COUNT(*) FROM plate_reads), "
                              "(SELECT COUNT(*) FROM person_dets)").fetchone()
    st.caption(f"{counts[0]} videos/images indexed · {counts[1]} plate reads · {counts[2]} person detections")
    if st.button("Clear saved result pictures", help="Search results and reports in the results folder. "
                 "They are made again when you search; nothing in the index is lost."):
        shutil.rmtree(RESULTS, ignore_errors=True)
        st.success("Cleared.")
