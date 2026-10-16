# Target Capture

Index a long video (or an image) once, then search it:

* **Plate search**: every license plate in the footage is read; search a number (exact, near-miss, part of the number, or wildcard) and get each time that car was seen, with clock time, vehicle make and model (e.g. Toyota Corolla), colour, and snapshots.
* **Arabic and Kurdish plates** (Iraq, Kurdistan Region): Arabic-Indic digits, Arabic letters, governorate and plate type. Detected automatically by default. Search with either digit style: `١٢٣٤٥` and `12345` find the same car.
* **Person search**: give one or more face photos; get how many times that person appears, and for each appearance the start and end time, duration, match confidence, clothing colours, and snapshots.

## Install

Python 3.10 to 3.13 (3.11 recommended).

### Windows

1. Python: `setup_windows.bat` finds it, or installs Python 3.11 for you if it is missing. (By hand: python.org, tick "Add python.exe to PATH".)
2. Install "Microsoft C++ Build Tools" (choose "Desktop development with C++"); InsightFace needs it.
3. Double-click `setup_windows.bat` once, then `run_windows.bat` whenever you want to use it.
   With an NVIDIA graphics card (GTX / RTX), click **Install graphics-card support** in the Settings tab once (or double-click `setup_gpu_windows.bat`): indexing gets several times faster.
   Settings also has **Open Target Capture when Windows starts** and **Start live monitoring when the page opens**, so after that you need no .bat file at all.
4. To get a new version, double-click `update_windows.bat` (needs Git for Windows). Your videos, index and results are kept.

Or by hand in Command Prompt, inside the `target-capture` folder:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

In PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
Each time you open a new window, run `.venv\Scripts\activate` again before using the tool.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The models (YOLO11n, the fast-alpr plate detector and OCR, EasyOCR Arabic, CLIP for car models, InsightFace `buffalo_l`) download automatically the first time you run the tool, about 1 GB in total.
If you have an NVIDIA GPU, install the CUDA build of `torch` and `onnxruntime-gpu` for a large speed-up (on Windows: Settings, **Install graphics-card support**, once). The left side of the web page says whether the GPU or the CPU is used.

## Web page: how to use it

### Start it

* **Windows:** double-click `run_windows.bat`. A black window opens and your browser opens the page by itself.
  If it doesn't, open the **Local URL** shown in the black window (usually `http://localhost:8501`).
* **Mac / Linux:** `streamlit run app.py`
* **From an iPad or phone at home:** keep the PC running and open the **Network URL** from the black window
  (e.g. `http://192.168.1.20:8501`) in Safari, on the same Wi-Fi. If Windows asks about the firewall, click Allow.

Keep the black window open while you use the page. Closing it stops Target Capture.

### Step 1: Upload & index (do this once per video)

1. Open the **1 · Upload & index** tab.
2. Either drag a video or photo into the upload box, **or** paste the full path of a file on this PC into the box
   below it (e.g. `D:\DVR\cam1.mp4`). For big DVR files, use the path: uploading hours of video through a browser is slow.
3. Leave **Read plates** and **Find people** ticked (untick one to go faster if you only need the other).
4. Optional: type the DVR clock time of the video's first frame (e.g. `2026-09-30 08:00:00`).
   Results then show real times like `08:42:15` instead of "2 h 42 min into the video".
5. Click **Start indexing** and wait for the progress bars to finish. A 1-minute clip takes under a minute on a normal PC;
   a 5-hour video can take a few hours. You can stop it and start again later: it continues where it stopped.

Under **Advanced** you can change:
* **Plate script:** *Auto* (default) reads English and Arabic plates. *English only* is faster if you only have English plates.
* **Car make and model:** untick to go faster if you don't need "Toyota Corolla" etc.
  Two recognizers (Settings tab): **fast** and **accurate** (1.7 GB, clearly better at telling models apart, but about
  8x slower, so meant for an NVIDIA GPU; "auto" uses it when there is one). Each car is looked at a few times as it
  comes closer and the looks are combined. The model year is not recognised; the list of models is car_models.txt.
* **Re-index from scratch:** tick this after an update to re-read an old video with the new features.

### Step 2: Search

**Find a plate** (tab 2)
* Type the plate and click **Search plate**. All of these work:
  `12345` (just the number), `١٢٣٤٥` (Arabic digits), `21 B 34567`, `12?45` (`?` = one unknown character), `12*` (`*` = anything after).
* **Allowed character differences** (default 1) still finds the car when one character was misread.
* Typing only part of a plate (3 characters or more, e.g. `21L1`) lists every plate that contains it under
  **Maybe the plate you want**; tick **View** to see one.
* Each result shows the time, how long the car was visible, vehicle make, model and colour, governorate,
  a picture of the frame, and a close-up of the plate.

**Find a person** (tab 3)
* Upload one or more clear photos of the person's face and click **Search person**.
  More photos (different angles) give better results.
* The answer is how many times the person was seen, and for each time: start and end, duration,
  how sure the match is (high / medium / low), clothing colours, and pictures.
* Too many wrong people? Move the **Match threshold** slider up. Missing sightings? Move it down a little.

**Find a car** (tab 4)
* Every vehicle the plate indexing saw is kept, **also when its plate could not be read**.
* Filter by make/model words (`toyota camry`), type, colour, *plate read / not read*, and a time range
  (video time `00:10:00`, or DVR clock `2026-09-30 08:00:00` when you gave the clock when indexing).
* Each result: the vehicle close-up, the whole frame, times, and the plate if one was read on it.

**Watch list**
* Add wanted plates (wildcards allowed) and wanted people (a face photo), with a note.
* Every video you index, and every live camera picture, is checked automatically. Matches show
  in the Watch list tab with pictures, and as a red banner at the top of every page (it refreshes itself).
* Adding an entry also checks the videos that are already indexed.

**Live cameras**
* Add a camera with its stream address (Hikvision: `rtsp://USER:PASS@IP:554/Streaming/Channels/101`,
  Dahua: `rtsp://USER:PASS@IP:554/cam/realmonitor?channel=1&subtype=0`) and click **Test connection**.
* A whole DVR / NVR: open **Add a whole DVR / NVR**, type its IP, user name, password and number of
  channels; every channel is added as a camera (switch off the ones you do not need).
* Click **Start live monitoring** (or tick **Start live monitoring when the page opens** in Settings). Each camera is analysed live,
  as the picture arrives: plates, people, vehicles and watch-list alerts appear a second or two after they
  happen. The tab shows what each camera sees now (with boxes), the latest plates and faces, and alerts.
  Everything found is also searchable in the other tabs, under one entry per camera per day.
* No recording is needed: the picture of each moment with a detection is kept in `live_frames\`.
  To also record the cameras for watching later, tick **Also record the cameras** in Settings.
* Pictures older than 7 days are deleted automatically, except days with a watch-list match.
* One graphics card shares its time between cameras: the status line shows how many pictures a second
  each camera gets. With many channels, switch off the ones that do not matter, or choose the Fast speed.
* The PC must stay on, and the cameras must be reachable from it (same network, or a VPN).

**Download results:** every search result, a plate opened with **View**, a person from All faces, the
watch-list matches and the whole All plates list can be downloaded as **Excel** (pictures in the sheet)
or **PDF**. The file is made when you click the button, so searches stay fast.

**Big videos:** files up to 20 GB can be uploaded (set in `.streamlit/config.toml`). For multi-hour DVR files,
typing the file's path is quicker because nothing is copied. Long videos do not need splitting.

### Step 3: Browse everything

**All faces**
* Every face in the video, grouped into people: **#1, #2, #3…**, each with a photo, how many times they were seen,
  and when they first appeared.
* Click **Show #N** to see every sighting of that person. Click **Close** to go back to the list.
* If two different people were put together, move the slider to the right. If one person appears twice in the list, move it left.

**Removing videos:** in the left sidebar, click the bin next to a video to remove it from the index, or
**Remove all videos** to start fresh. Your original video files are not deleted (only copies the app made:
uploads and repaired DVR files), and what you taught in the Teach tab is kept. Command line:
`python -m targetcapture remove --all`.

**All plates**
* **Read plates:** one row per car pass, with times, vehicle, and the words printed on the plate around the
  number (`العراق`, `السليمانية`, `SUL`, `ERB` ...) in **region / plate words**. Type in the filter box
  (e.g. `corolla`, `white toyota`, `Erbil`) to narrow it down. Tick **View** at the end of a row to open that
  plate like a search result: the best capture of the car and a second capture taken after it.
* **Plates that could not be read:** pictures of plates that were found but not read (other alphabets,
  blur, too small). Click **Open** under one to see the whole frame.

### Teach: correct the system

Use this when a plate was not read, or was read wrongly.

1. Pick what to check: **Plates it could not read**, **Check read plates** (least sure first),
   or **Correct a specific plate** (type the plate as the All plates table shows it).
2. Look at the close-up and the full frame, type the correct text (Arabic or English digits), and click **Save**.
   **No plate** removes a false detection; **Skip** moves to the next one.
3. After saving you may see:
   * **Also fix N other reads that were read as "…"**: the same reading mistake elsewhere. One click fixes them all.
   * **These look like the same plate**: captures that look similar. Tick only the ones where the number really matches.

What the system does with your answers:
* The corrected plate is searchable straight away, and shows "typed by you" in the All plates table.
* If the reader makes the **same mistake** again (in new videos too), it is corrected automatically and marked "learned".
* Similar-looking plates are only ever **suggested**, never changed by themselves: plates of the same style with
  different numbers look almost identical to a picture comparison, so only you can confirm the number.
* **Export** (at the bottom of the tab, or `python -m targetcapture export-taught`) saves your corrected plates
  as pictures plus `labels.csv`. With a few hundred of them, the plate reader itself can be retrained on your
  cameras' plates, which is the real long-term accuracy gain.

A second plate reader trained on Iraqi plates (`targetcapture/models/iraq_plate_ocr.onnx`) runs next to the general one.
It takes over on two-line plates the general reader garbles, and when the general read is too weak; single-line and
foreign plates stay with the general reader. To retrain it on your own corrected plates (needs the CPU or GPU for an hour or so):

```bat
.venv\Scripts\python -m targetcapture export-taught taught
.venv\Scripts\python tools\train_plate_ocr.py --resume targetcapture\models\iraq_plate_ocr.pt --real taught --minutes 60
```

### Best results tab

The **Best results** tab (also in `TIPS.md`) explains how to get the most out of your cameras, settings and searches.

### Settings tab

* **Version** of this copy and the date of the installed update. **Check for updates** lists what is new on GitHub;
  **Update now** runs update_windows.bat in its own window, closes the page and opens it again when done.
* **Defaults** for indexing, searching, the watch list, live cameras and the largest upload. They are saved in
  `settings.json`, which updates keep.
* **This computer** (GPU or CPU, library versions) and **Storage** (size of the index, uploads, recordings and results,
  free disk space).

### Where results are saved

Every search also saves a report in the `results` folder: `report.html` (open it in a browser to see or print it),
plus `results.csv` (opens in Excel) and `results.json`.

### DVR footage: fetch recordings straight from the DVR / NVR

Tab **DVR footage** starts with **Your DVRs / NVRs**: the DVRs / NVRs found on this PC's network (the
same search as Hikvision SADP, Dahua ConfigTool and ONVIF tools; **Search the network again** repeats
it, **Also show single IP cameras** adds cameras), plus the ones you saved here or added in Live
cameras, with their make, address, model, serial number and how many cameras are online. A DVR found
on the network asks for its password once (tick **Save these DVR details**). Click one to see
**its cameras** (read from the DVR / NVR itself, see below); click a camera to download only its
channel. **Add a new DVR / NVR** opens an empty form. Then pick the channels, a date and a time range,
and click **Download and index**.

* The camera list comes from the DVR / NVR's own settings (its Camera Management page): channel,
  name, online / offline / no video, analog or IP, and for IP cameras their IP address, port, model,
  protocol, serial number and firmware. **Read the cameras again** refreshes it; **List the DVR's
  cameras** in the form reads it for a DVR not saved yet.
* **Check what is recorded** asks the DVR which minutes have recordings, so empty hours are skipped.
* The range is cut into pieces (10 minutes by default). The next piece downloads while one is
  indexed, and each piece is searchable as soon as it is done, with the DVR's own clock time.
* Download uses the DVR's web download (Hikvision ISAPI, Dahua loadfile.cgi), which is faster than
  real time. If the DVR refuses it, RTSP playback is copied instead (real-time speed only).
* It runs in its own background process, so it continues when the page is closed. Progress and a
  Stop button are under **Downloads** in the same tab.
* Several pieces download at the same time (**Download at the same time**, 2 by default); each piece
  shows its size, MB/s and time left, and the job shows the total speed.
* **Save downloads in** picks the folder (default `dvr_videos/`). Tick **Delete each video after
  indexing** to save disk space: the pictures of plates and faces are kept, so results still show.
  **Delete indexed DVR downloads** (here and in Settings > Storage) removes older downloads.
* **Save these DVR details** keeps the DVR's name, address and login; pick it next time under
  **Saved DVR / NVR**.
* **Index videos downloaded with the DVR's own web page**: point it at the folder you download into;
  every video is indexed, and with **Keep watching** each new file as soon as its download finishes.
  The time in the file name is used as the clock time.

### Use it from other devices (phone, laptop, another office)

The work always runs on the PC with the graphics card; other devices only open the page.
Settings > **Use from other devices** shows the address to open. On the same network it works at once.
From anywhere, install [Tailscale](https://tailscale.com/download) (free) on the PC and on the phone or
laptop, signed in with the same account: the PC's 100.x.x.x address then shows there, and only your
own devices can reach it. Set a **page password** first: other devices are asked for it (this PC is
not). Do not open the port on the router to the whole internet.

## Command line

```bash
# 1. Index (both plates and people by default; --plates or --persons for only one)
python -m targetcapture index /videos/cam1_2026-09-30.mp4 --clock-start "2026-09-30 08:00:00"

# English and Arabic plates are told apart automatically (--plate-script auto)

# 2. Search plates
python -m targetcapture find-plate 12ABC345          # exact or 1 character off
python -m targetcapture find-plate "12?BC*"          # partial plate: ? = one char, * = anything
python -m targetcapture find-plate 34567             # only the number: ignores governorate code and letter
python -m targetcapture find-plate "١٢٣٤٥"           # Arabic-Indic digits work too
python -m targetcapture plates                       # list every plate seen
python -m targetcapture plates --vehicle "white corolla"   # filter by make, model, colour or governorate

# 3. Search a person
python -m targetcapture find-person suspect1.jpg suspect2.jpg

python -m targetcapture media                        # what is indexed

# 4. Live cameras (also from the web page)
python -m targetcapture live --add "Gate" rtsp://admin:pass@192.168.1.64:554/Streaming/Channels/101
python -m targetcapture live                         # analyse all cameras live until Ctrl+C
```

Each search writes `results/<query>/report.html` (snapshots and details), plus `results.json` and `results.csv`.
See `examples/` for two finished reports from test clips.

Indexing saves progress every 10 seconds; press Ctrl-C at any time and run the same command again to resume.
Several videos can go into one index (`--db` picks the file, default `targetcapture.db`), and searches cover all of them.

### Useful options

| Option | Default | What it does |
|---|---|---|
| `index --speed` | (none) | `fast` (plates 3, people 2 frames per second), `balanced` (4 / 3) or `thorough` (6 / 4). Overrides the two rates below. The web page uses the Speed choice (default Fast). |
| `index --plate-fps` | 4 | Frames per second checked for plates. Raise for fast traffic. |
| `index --person-fps` | 3 | Frames per second checked for people. |
| `index --vehicle-crops` | off | Also reads plates inside each detected vehicle. Slower; helps on wide or 4K views where plates are tiny. |
| `index --plate-script` | auto | `auto` reads English plates with the Latin reader and Arabic plates with the Arabic reader; `latin` is English only (fastest); `arabic` uses the Arabic reader for every plate. |
| `index --no-car-model` | off | Skip make/model recognition (faster). The list of models is `car_models.txt`; edit it to match your roads. |
| `index --no-motion-skip` | off | By default frames where nothing moved are skipped (a still frame is still checked every 5 s). |
| `find-plate --max-dist` | 1 | How many characters may differ. |
| `find-plate --min-reads` | 1 | Ignore one-off reads (OCR noise). |
| `find-person --threshold` | 0.40 | Face similarity needed. Lower finds more but risks wrong people; check "near misses" in the report. |

## Speed

Measured on a 4-core CPU with no GPU: person indexing processed about 8 analysed frames per second on a 768x432 clip with 1 to 3 people in view, and plate indexing took 6 s for a 1-minute 720p clip.
At those rates a 5-hour busy 1080p street video would take a few hours on CPU; a GPU is typically 5 to 20 times faster. Searching an existing index takes seconds.

After each video, the page (and the command line) shows how fast it went, as "× real time" with the time for
10 hours of video, and which step took the time. Indexing 10 hours in 1 hour means 10× real time; reaching it
needs an NVIDIA GPU (setup_gpu_windows.bat) and the Fast speed. On a 4-core CPU with no GPU it is about 3 to 8×
real time depending on how busy the road is.

Recent speed-ups (same results, less work):

* Each video is decoded once for plates and people (it used to be decoded twice), and one detector run finds
  both people and vehicles in a frame (it used to be two runs). The plate reader and the face reader work side by
  side in their own threads while the next frames are decoded. On a 4-core CPU this made indexing 1.4 to 1.7×
  faster at the same settings.
* Speed choice: Fast looks at 3 frames a second for plates and 2 for people and works out each car's model
  from 2 views; most plates and faces are still seen several times as they pass.
* The car model recognizer runs in half precision on an NVIDIA GPU.

* Models are loaded once and reused for every video in a batch and for every live camera piece.
* Faces are read only when useful: the first 3 per person, then every 0.6 s or when the person gets clearly bigger. On a busy town clip this halved person indexing time (478 s to 238 s) with identical search results.
* The car model is worked out once per car (and again only if the car gets much bigger), not on every frame.
* On an NVIDIA GPU the detectors run in half precision.
* The video is read in the background while the detectors work, and decoded on the graphics card when possible
  (Settings tab; falls back to normal decoding by itself). Pictures of results are fetched with the same decoder the
  video was indexed with, so they match exactly.
* Long videos need no splitting: indexing saves its place every 10 seconds and can be stopped and continued.

Plate pictures: the capture shown for each car is the clearest one (reader confidence, plate size and sharpness, so
motion-blurred frames lose), and close-ups are enlarged with a high-quality filter, contrast-evened, lightly sharpened
and de-interlaced when the DVR video is interlaced. This makes readable plates easier to read; it cannot recover
characters the camera never captured.
* Optional: `--skip-empty-road` (or the Advanced checkbox "Faster: skip plate reading when no vehicle is in view") skips plate reading on frames with no vehicle. It is much faster on quiet roads but can miss plates on cars the detector does not see, so it is off by default.

## Run the tests

```bash
python -m pytest tests                                     # matching logic, no models needed
python tests/make_plate_sample.py car.jpg demo.mp4         # synthetic clip: one car passes twice
```

## Responsible use

Identifying people in footage is regulated in many places (for example GDPR in the EU, and biometric laws in several US states).
Make sure you have a lawful basis for processing the footage and for face matching, keep the index and reports secured, and delete them when no longer needed.
Treat matches as leads for a human to verify, never as proof of identity.

## Licences of the bundled models

* Ultralytics YOLO is AGPL-3.0 (a commercial licence is available from Ultralytics).
* The InsightFace `buffalo_l` face models are for non-commercial research use only. For commercial use, swap in a commercially licensed face model (see DESIGN.md).
* fast-alpr is MIT.
