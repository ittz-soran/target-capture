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
If you have an NVIDIA GPU, install the CUDA build of `torch` and `onnxruntime-gpu` for a large speed-up.

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
* **Re-index from scratch:** tick this after an update to re-read an old video with the new features.

### Step 2: Search

**Find a plate** (tab 2)
* Type the plate and click **Search plate**. All of these work:
  `12345` (just the number), `١٢٣٤٥` (Arabic digits), `21 B 34567`, `12?45` (`?` = one unknown character), `12*` (`*` = anything after).
* **Allowed character differences** (default 1) still finds the car when one character was misread.
* Each result shows the time, how long the car was visible, vehicle make, model and colour, governorate,
  a picture of the frame, and a close-up of the plate.

**Find a person** (tab 3)
* Upload one or more clear photos of the person's face and click **Search person**.
  More photos (different angles) give better results.
* The answer is how many times the person was seen, and for each time: start and end, duration,
  how sure the match is (high / medium / low), clothing colours, and pictures.
* Too many wrong people? Move the **Match threshold** slider up. Missing sightings? Move it down a little.

### Step 3: Browse everything

**All faces** (tab 4)
* Every face in the video, grouped into people: **#1, #2, #3…**, each with a photo, how many times they were seen,
  and when they first appeared.
* Click **Show #N** to see every sighting of that person. Click **Close** to go back to the list.
* If two different people were put together, move the slider to the right. If one person appears twice in the list, move it left.

**All plates** (tab 5)
* **Read plates:** a table of every plate read, with times, vehicle and details. Type in the filter box
  (e.g. `corolla`, `white toyota`, `Erbil`) to narrow it down.
* **Plates that could not be read:** pictures of plates that were found but not read (other alphabets,
  blur, too small). Click **Open** under one to see the whole frame.

### Teach: correct the system (tab 6)

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

### Best results tab

The **Best results** tab (also in `TIPS.md`) explains how to get the most out of your cameras, settings and searches.

### Where results are saved

Every search also saves a report in the `results` folder: `report.html` (open it in a browser to see or print it),
plus `results.csv` (opens in Excel) and `results.json`.

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
```

Each search writes `results/<query>/report.html` (snapshots and details), plus `results.json` and `results.csv`.
See `examples/` for two finished reports from test clips.

Indexing saves progress every 10 seconds; press Ctrl-C at any time and run the same command again to resume.
Several videos can go into one index (`--db` picks the file, default `targetcapture.db`), and searches cover all of them.

### Useful options

| Option | Default | What it does |
|---|---|---|
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
