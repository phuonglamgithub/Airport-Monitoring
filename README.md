# Checkin_Monitor

AI-powered passenger monitoring system for airport check-in areas.  
Built with **Flask**, **Ultralytics YOLO**, and **ByteTrack**.

---

## Features

### Detection & class mapping

| Feature | Detail |
|---|---|
| **Dynamic class resolution** | Reads `model.names` from the loaded checkpoint at runtime. Class IDs are never hardcoded — the engine maps them by name regardless of dataset. |
| **COCO fallback** | If `model.names` cannot be resolved, falls back to standard COCO indices (`person=0`, `backpack=24`, `handbag=26`, `suitcase=28`, `cat=15`, `dog=16`). |
| **YOLO26 / Open Images V7** | Supports multi-hundred-class checkpoints. Broad name aliases (`luggage and bags`, `duffel bag`, `human body`, etc.) are matched to the four UI labels. |
| **Passenger classes** | Person · Luggage (Backpack / Handbag / Suitcase) · Pet · Hurry Up/Busy |
| **Per-class confidence** | Luggage uses a lower confidence threshold (0.15) to improve recall on small or partially occluded bags. Person and Pet use the threshold set in the UI. |
| **Sub-type labels** | Bounding box labels show the specific checkpoint class name (e.g. `Suitcase 0.82`) instead of the generic group label. |

### Behavioral analysis

| Feature | Detail |
|---|---|
| **Hurry Up / Busy** | Derived from ByteTrack trajectory. Velocity is computed over the last 12 positions. Persons exceeding `HURRY_VEL = 55 px/s` are flagged and relabelled. |
| **Fallback centroid tracker** | When YOLO or ByteTrack does not return a stable track ID (e.g. short clips, some Ultralytics versions), a centroid-based tracker assigns temporary IDs so that crossing counting still works. |

### Counting

| Feature | Detail |
|---|---|
| **Cumulative IN / OUT** | Total IN and Total OUT increase only when a tracked object crosses the counting line. They are never reset by zone changes. |
| **CrossingTracker** | Margin zone (±`LINE_MARGIN = 25 px`) around the line suppresses oscillation. Per-ID cooldown (`CNT_COOLDOWN = 2.0 s`) prevents double-counting the same crossing event. |
| **Counting axis** | Vertical line (`count_axis = "x"`) — recommended for check-in cameras where passengers move left/right. Horizontal line (`count_axis = "y"`) — for cameras looking down a corridor. |
| **Zone occupancy** | Separate from cumulative totals. Counts how many tracked objects are currently on the IN side vs the OUT side of the line. Displayed in the stats but does **not** modify Total IN / Total OUT. |
| **Busy count** | Number of persons currently flagged as fast-moving. Returned in `/api/stats` as `busy_count`. |

### Counting presets (6)

Set in the UI via the **Counting Ratio & Direction** dropdown. Format: `count_frac:count_in_side`.

| Value | Line position | Left / Top = | Right / Bottom = |
|---|---|---|---|
| `0.30:left` | 30 % from left | IN | OUT |
| `0.30:right` | 30 % from left | OUT | IN |
| `0.50:left` | Centre | IN | OUT |
| `0.50:right` *(default)* | Centre | OUT | IN |
| `0.70:left` | 70 % from left | IN | OUT |
| `0.70:right` | 70 % from left | OUT | IN |

### Server & stream

| Feature | Detail |
|---|---|
| **Soft Stop** | `engine.stop()` stops inference and releases the camera, but Flask and the `/video_feed` MJPEG endpoint stay alive. The browser keeps its connection open and the user can press Start again without a page refresh. |
| **MJPEG reconnect** | `app.js` re-requests `/video_feed?t=<timestamp>` on tab switch and after Stop to prevent the browser from serving a stale or broken stream. |
| **Thread-safe camera** | `_cap_lock` ensures the `cv2.VideoCapture` handle is never released mid-read. A `threading.Event` (`_stop_event`) signals the detection thread to exit cleanly. |
| **Session persistence** | Each stopped session is appended to `logs/sessions.json`. The Statistics Dashboard aggregates all past sessions without double-counting the live session. |

---

## Operation modes

| Mode | Bounding boxes | Track IDs | Counting line | IN / OUT totals |
|---|---|---|---|---|
| Detection Only | ✓ | — | — | — |
| Counting Mode | — | ✓ | ✓ | ✓ |
| Detection & Counting | ✓ | ✓ | ✓ | ✓ |

---

## Quick start

1. **Counting Line** — choose axis (Vertical / Horizontal) and a ratio+direction preset
2. **Operation Mode** — Detection Only · Counting Mode · Detection & Counting
3. **Model Selection** — pick a checkpoint from the dropdown or upload a `.pt` file → **Load Model**
4. **Video Source** — Webcam or uploaded video file
5. **Classes** — tick Person, Luggage, Hurry Up/Busy, Pet
6. **Start Detection**

Pressing **Stop** pauses inference but keeps the page and stream alive. Press **Start** again to resume.

---

## Tunable constants (`detector.py`)

```python
HURRY_VEL    = 55    # px/s  — speed threshold for Hurry Up/Busy label
LINE_MARGIN  = 25    # px    — neutral band on each side of the counting line
CNT_COOLDOWN = 2.0   # s     — minimum interval between two counts for the same track ID
CLASS_CONF   = {
    "Person":  None,   # use confidence from UI slider
    "Luggage": 0.15,   # lower threshold to improve recall for bags
    "Pet":     None,
}
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Main page |
| GET | `/video_feed` | MJPEG stream (always alive) |
| POST | `/api/load_model` | Load a checkpoint |
| POST | `/api/start` | Start detection |
| POST | `/api/stop` | Soft stop |
| GET | `/api/stats` | Live stats, counts, timeline |
| GET | `/api/sessions` | All saved sessions |
| POST | `/api/clear_sessions` | Delete session history |
| POST | `/api/upload_video` | Upload a video file |
| POST | `/api/upload_model` | Upload a custom `.pt` file |
| GET | `/api/list_models` | List checkpoints in `models/` |

---

## Tech stack

`YOLOv11 / YOLOv12 / YOLO26` &nbsp;·&nbsp; `ByteTrack` &nbsp;·&nbsp; `OpenCV` &nbsp;·&nbsp; `Flask` &nbsp;·&nbsp; `Chart.js` &nbsp;·&nbsp; `NVIDIA CUDA`
