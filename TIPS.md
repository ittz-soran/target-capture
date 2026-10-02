## How to get the best results

The system can only find what the camera recorded clearly. Most misses come from the video, not the search.

### 1. Give it the best video you have

* **Export the original recording** from the DVR/NVR (the *main stream*, full resolution), not the *sub stream* and
  not a phone recording of the screen. Re-compressed or WhatsApp-forwarded videos lose the detail plates and faces need.
* **1080p or higher** works best. 720p is fine for close cameras; at 480p plates and faces are often unreadable.
* **Index the video file directly**: on the PC, paste its path in tab 1 instead of uploading it through the browser.
* **Raw DVR exports** (`.h264`, `.265`, some `.ts`/`.mp4`) often cannot jump to an exact moment. Target Capture checks this
  and, when needed, first copies the video into `fixed_videos\` as a normal `.mp4` (no quality loss, takes seconds).
  If a red box ever sits on the road instead of the plate, tick *Re-index from scratch* for that video.

### 2. What the camera needs to see

| | Plates | Faces |
|---|---|---|
| Size in the picture | plate at least **100 pixels wide** | face at least **60 pixels tall** |
| Angle | within about **30°** of straight on | looking roughly towards the camera |
| Light | IR at night is fine for plates | needs light on the face; IR night video works poorly |
| Motion | fast shutter (DVR setting) avoids blurred plates | people walking towards the camera are best |

If you can adjust your cameras: a narrow view of an entrance or a gate (cars slowing down, people walking in)
reads far better than a wide view of a whole street.

### 3. Settings in tab 1 (Upload & index → Advanced)

* **Plate script: Auto** reads both English and Arabic plates. Use *English only* only if all plates are English (faster).
* **Frames per second for plates:** 4 is good for parking areas and slow traffic. Use **6–10 for fast roads**.
* **Frames per second for people:** 3 is fine. Raise it for crowded places where people pass quickly.
* **Also read plates inside each vehicle:** turn on for wide or 4K views where plates look tiny.
* **DVR clock:** type the DVR time of the first frame so every result shows the real time.
* Untick **Read plates** or **Find people** if you only need one; indexing is about twice as fast.

### 4. Searching plates

* Start with **just the number** (e.g. `12345`). It matches even if the governorate code or letter was missed.
* If nothing is found, raise **Allowed character differences** to 2, or use wildcards: `12?45`, `123*`.
* Arabic digits (`١٢٣٤٥`) and English digits (`12345`) find the same plate.
* Check **All plates → Plates that could not be read**: the car may be there with an unreadable plate.
* Use the vehicle filter (`white corolla`, `Erbil`) when you only know the car.

### 5. Searching people

* Use **2–4 clear photos** of the face from different angles, well lit, face filling most of the photo.
  A passport-style photo plus one side view is ideal.
* **Too many wrong people?** Move the threshold up (0.45–0.50). **Missing sightings?** Move it down (0.35).
  Results marked *low* confidence should always be checked by eye.
* No photo? Open **All faces**, find the person there, and click **Show** to get all their sightings.
* People are found by **face only**. Someone who never faces the camera will not be found.

### 6. Teach it (the more you correct, the better it gets)

* After indexing, spend a few minutes in **Teach → Plates it could not read** and **Check read plates**.
* Each correction is searchable at once and fixes the same reading mistake in future videos.
* Use **Export** regularly. With a few hundred corrected plates, the plate reader can be retrained on
  *your* cameras' plates, the biggest accuracy improvement available.

### 7. Speed

* A PC with an **NVIDIA graphics card** indexes many times faster than the processor alone. Run `setup_gpu_windows.bat` once so it is used; the left side of the page then shows "GPU: <card name>".
* Long videos: start indexing in the evening and leave the window open. If it stops, start it again:
  it continues where it left off.
* Searching an already indexed video takes seconds. You never need to index the same video twice
  (except after an update that adds new features, using *Re-index from scratch*).

### 8. Use it responsibly

Results are **leads for a person to check**, not proof. Face matching of people in footage is regulated in many
places: only use it where you are allowed to, and keep the videos, index and reports secure.
