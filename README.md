# fixed_checkin_monitor_complete

AI Check-in Monitor demo using Flask + Ultralytics YOLO.

## Main features

- YOLO inference with COCO or YOLO26.
- Dynamic class mapping from `model.names`, not hard-coded COCO IDs only.
- Person / Luggage / Pet detection.
- Busy / Fast-moving passenger estimation from tracked movement speed.
- ObjectCounter-style Total IN / Total OUT using crossing-line logic.
- RegionCounter-style current IN-side / OUT-side occupancy.
- 6 counting presets:
  - Left 30% / Right 70% — LEFT = IN, RIGHT = OUT
  - Left 30% / Right 70% — LEFT = OUT, RIGHT = IN
  - Left 50% / Right 50% — LEFT = IN, RIGHT = OUT
  - Left 50% / Right 50% — LEFT = OUT, RIGHT = IN
  - Left 70% / Right 30% — LEFT = IN, RIGHT = OUT
  - Left 70% / Right 30% — LEFT = OUT, RIGHT = IN
- Soft Stop: pressing Stop pauses detection but keeps the Flask app and video endpoint alive.
- Statistics dashboard with Chart.js.
- Video frame is constrained in the UI while preserving the original video/camera aspect ratio.

## Folder structure

```text
fixed_checkin_monitor_complete/
├── app.py
├── detector.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── models/
├── uploads/
└── logs/
```

## Run

```bash
cd fixed_checkin_monitor_complete
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Counting note

Total IN / Total OUT is cumulative and only changes when a tracked object crosses the counting line. Current zone occupancy is separate and is not used to overwrite the cumulative totals.
