# ============================================================
# detector.py  —  Detection + Crossing Counter + Zone Monitoring Engine
# Classes: Person · Luggage · Hurry Up/Busy · Pet
# ============================================================

import cv2
import numpy as np
import threading
import time
import json
import os
from collections import defaultdict, deque
from datetime import datetime

# Default COCO IDs are kept only as a fallback.
# For YOLO26 / Open Images V7, the engine resolves class IDs dynamically
# from model.names instead of assuming COCO indices.
CLASS_COCO_FALLBACK = {
    "Person":        [0],
    "Luggage":       [24, 26, 28],   # backpack, handbag, suitcase
    "Pet":           [15, 16],       # cat, dog
    "Hurry Up/Busy": [],
}

# Open Images V7 / COCO class-name aliases used by the business labels in the UI.
# The checkpoint's model.names is the source of truth; this list only maps
# detailed dataset labels to demo labels: Person / Luggage / Pet.
CLASS_ALIASES = {
    "Person": {
        "person", "man", "woman", "boy", "girl", "human", "human body",
        "pedestrian", "people",
    },
    "Luggage": {
        "luggage", "luggage and bags", "suitcase", "backpack", "handbag",
        "bag", "briefcase", "purse", "duffel bag", "duffle bag",
        "travel bag", "baggage",
    },
    "Pet": {
        "pet", "dog", "cat", "rabbit", "hamster", "guinea pig", "bird",
    },
    "Hurry Up/Busy": set(),
}

CLASS_COLOR = {
    "Person":        (219, 152,  52),
    "Luggage":       ( 76, 153,   0),
    "Pet":           (182,  89, 155),
    "Hurry Up/Busy": ( 50,  76, 226),
}

# ── Confidence riêng từng class ────────────────────────────
# Luggage (balo, túi xách) cần threshold thấp hơn Person
CLASS_CONF = {
    "Person":        None,    # dùng conf từ UI
    "Luggage":       0.15,    # override: balo/túi xách khó detect hơn
    "Pet":           None,
    "Hurry Up/Busy": None,
}

# ── Tên chi tiết cho sub-type Luggage ─────────────────────
# COCO fallback only. For Open Images V7 / YOLO26, labels are read
# directly from model.names, for example Suitcase / Backpack / Handbag.
LUGGAGE_NAMES = {
    24: "Backpack",
    26: "Handbag",
    28: "Suitcase",
}

HURRY_VEL  = 55
LINE_FRAC  = 0.50  # default: line at 50% of frame width/height

# ── Counting parameters ────────────────────────────────────
LINE_MARGIN  = 25    # px vùng trung lập quanh đường line (tránh dao động)
CNT_COOLDOWN = 2.0   # giây tối thiểu giữa 2 lần đếm cùng 1 track ID


class CrossingTracker:
    """
    Theo dõi chiều vượt line cho từng track ID.
    Dùng margin zone + cooldown để tránh đếm sai.

    Logic:
      - Chia frame thành 3 vùng theo chiều dọc:
          ABOVE  : cy < line_y - MARGIN
          ZONE   : cy trong margin (không trigger)
          BELOW  : cy > line_y + MARGIN
      - Chỉ đếm khi side thay đổi ABOVE ↔ BELOW
        (bỏ qua trạng thái ZONE ở giữa)
      - Cooldown tránh đếm lại cùng người trong thời gian ngắn
    """

    def __init__(self, margin: int = LINE_MARGIN,
                 cooldown: float = CNT_COOLDOWN):
        self.margin   = margin
        self.cooldown = cooldown
        self._side    = {}      # tid → 'above' | 'below'
        self._last_t  = {}      # tid → timestamp lần đếm cuối

    def update(self, tid, pos: int, line_pos: int, now: float, in_side: str = "after") -> str | None:
        """
        Trả về 'IN', 'OUT', hoặc None.

        pos có thể là cx nếu dùng vertical line, hoặc cy nếu dùng horizontal line.
        in_side:
          - 'before': phía trước line là vùng IN
              x-line: LEFT  = IN; y-line: TOP    = IN
          - 'after' : phía sau line là vùng IN
              x-line: RIGHT = IN; y-line: BOTTOM = IN

        Như vậy user có thể chọn:
          LEFT = IN / RIGHT = OUT  hoặc  LEFT = OUT / RIGHT = IN.
        """
        if pos < line_pos - self.margin:
            new_side = 'before'
        elif pos > line_pos + self.margin:
            new_side = 'after'
        else:
            return None   # trong vùng trung lập

        prev_side = self._side.get(tid)
        self._side[tid] = new_side

        if prev_side is None or prev_side == new_side:
            return None

        if now - self._last_t.get(tid, 0) < self.cooldown:
            return None

        self._last_t[tid] = now

        if prev_side == 'before' and new_side == 'after':
            return 'IN' if in_side == 'after' else 'OUT'
        if prev_side == 'after' and new_side == 'before':
            return 'IN' if in_side == 'before' else 'OUT'
        return None

    def reset(self):
        self._side.clear()
        self._last_t.clear()


class DetectionEngine:
    def __init__(self):
        self.model      = None
        self.model_name = None
        self.running    = False
        self.cap        = None
        self._lock      = threading.Lock()
        self._frame     = None
        self._thread    = None
        self._cap_lock  = threading.Lock()
        self._stop_event = threading.Event()
        self._vel_buf   = {}
        self._crossing  = CrossingTracker()   # thay thế _line_prev/_in_ids/_out_ids

        # Fallback tracker dùng khi YOLO/ByteTrack không trả track_id.
        # Nếu không có ID ổn định thì IN/OUT gần như không thể đếm chính xác.
        self._fallback_tracks = defaultdict(dict)
        self._next_fallback_id = 1

        self.stats      = self._blank()
        self.config     = {}

    @staticmethod
    def _blank():
        return {
            "running": False, "fps": 0.0, "frame_no": 0,
            "current": {},
            "in_counts":  defaultdict(int),
            "out_counts": defaultdict(int),
            "fps_hist": deque(maxlen=300),
            "timeline": deque(maxlen=600),
            "current_in_zone": {},
            "current_out_zone": {},
            "busy_count": 0,
            "start_time": None,
        }

    # ── model ─────────────────────────────────────────────
    def load_model(self, mtype, path):
        try:
            from ultralytics import YOLO
            if mtype == "pretrained":
                full = os.path.join("models", path)
                self.model = YOLO(full if os.path.exists(full) else path)
            else:
                self.model = YOLO(path)
            self.model_name = path

            dataset = self._infer_dataset_name()
            cmap = self._class_map_summary()
            msg = f"Loaded: {path} | Dataset: {dataset} | {cmap}"
            return {"success": True, "message": msg}
        except Exception as e:
            return {"success": False, "message": str(e)}

    @staticmethod
    def _norm_name(name):
        return (str(name).strip().lower()
                .replace("&", "and")
                .replace("_", " ")
                .replace("-", " ")
                .replace("/", " "))

    def _model_names(self):
        """Return {class_id: class_name} from the loaded checkpoint."""
        if not self.model:
            return {}
        names = getattr(self.model, "names", None)
        if names is None:
            try:
                names = self.model.model.names
            except Exception:
                names = {}
        if isinstance(names, dict):
            out = {}
            for k, v in names.items():
                try:
                    out[int(k)] = str(v)
                except Exception:
                    continue
            return out
        if isinstance(names, (list, tuple)):
            return {i: str(v) for i, v in enumerate(names)}
        return {}

    def _infer_dataset_name(self):
        names = self._model_names()
        n = len(names)
        if n >= 500:
            return f"Open Images V7-like ({n} classes)"
        if n == 80:
            return "COCO (80 classes)"
        if n:
            return f"Custom / unknown ({n} classes)"
        return "Unknown"

    def _resolve_class_ids(self, selected):
        """
        Map UI labels to checkpoint class IDs.

        This is the key change for YOLO26 trained on Open Images V7:
        the app no longer assumes COCO IDs such as person=0 or suitcase=28.
        Instead, it reads model.names and finds class IDs by class names.
        """
        names = self._model_names()
        resolved = defaultdict(list)

        if names:
            for cid, cname in names.items():
                norm = self._norm_name(cname)
                for ui_cls, aliases in CLASS_ALIASES.items():
                    if ui_cls == "Hurry Up/Busy":
                        continue
                    if norm in aliases:
                        resolved[ui_cls].append(cid)

        # Fallback for standard COCO checkpoints or checkpoints without usable names.
        if not any(resolved.values()):
            for ui_cls, ids in CLASS_COCO_FALLBACK.items():
                resolved[ui_cls].extend(ids)

        # Hurry Up/Busy is derived from Person speed, so it needs Person IDs.
        if "Hurry Up/Busy" in selected:
            resolved["Hurry Up/Busy"] = list(resolved.get("Person", []))

        return {k: sorted(set(v)) for k, v in resolved.items()}

    def _class_map_summary(self):
        cmap = self._resolve_class_ids(["Person", "Luggage", "Pet", "Hurry Up/Busy"])
        parts = []
        for k in ("Person", "Luggage", "Pet"):
            parts.append(f"{k}={len(cmap.get(k, []))} IDs")
        return "Mapping: " + ", ".join(parts)

    def _class_name(self, cid):
        return self._model_names().get(int(cid), str(cid))

    # ── start / stop ──────────────────────────────────────
    def start(self, cfg):
        """
        Start/re-start detection safely.
        Important for demo use: after pressing Stop, users can press Start again
        without refreshing the browser or restarting Flask.
        """
        if self.running:
            return {"success": False, "message": "Already running"}
        if not self.model:
            return {"success": False, "message": "Load a model first"}

        # Make sure a previous Stop has fully finished.
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=1.0)

        self.config = cfg
        self.stats  = self._blank()
        self.stats.update({"running": True,
                           "start_time": datetime.now().isoformat()})
        self._vel_buf = {}
        self._crossing.reset()
        self._fallback_tracks.clear()
        self._next_fallback_id = 1
        self._stop_event.clear()

        src = cfg.get("source", "webcam")
        cap = cv2.VideoCapture(0 if src == "webcam" else cfg.get("video_path", ""))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            self.stats["running"] = False
            cap.release()
            return {"success": False, "message": "Cannot open video source"}

        with self._cap_lock:
            # Release any old handle only after the new source is ready.
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = cap

        self.running  = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return {"success": True, "message": "Started"}

    def stop(self):
        """
        Soft stop: stop detection and release the camera/video safely, but keep
        Flask and the MJPEG endpoint alive. The browser stays connected and the
        user can press Start again immediately.
        """
        was_running = self.running or self.stats.get("running", False)
        self.running = False
        self.stats["running"] = False
        self._stop_event.set()

        th = self._thread
        if th and th.is_alive() and th is not threading.current_thread():
            th.join(timeout=1.5)

        with self._cap_lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

        if was_running:
            self._save_session()
        self._thread = None
        return {"success": True, "message": "Stopped. You can press Start again."}

    # ── main loop ─────────────────────────────────────────
    def _loop(self):
        conf  = float(self.config.get("confidence",  0.25))
        iou   = float(self.config.get("iou",         0.45))
        skip  = max(1, int(self.config.get("frame_skip", 3)))
        mode  = self.config.get("mode", "detection")
        sel   = self.config.get("classes", ["Person"])
        track = (mode in ("counting", "detection_counting"))

        # Resolve class IDs from the loaded checkpoint.
        # COCO and Open Images V7 use different class indices, so hardcoded
        # COCO IDs would fail with YOLO26/Open Images checkpoints.
        class_map = self._resolve_class_ids(sel)

        ids = []
        rev = {}
        for c in sel:
            if c == "Hurry Up/Busy":
                continue
            for i in class_map.get(c, []):
                if i >= 0:
                    ids.append(i)
                    rev[i] = c

        # Hurry Up/Busy is derived from Person tracking speed.
        # Therefore Person must still be detected even when only Hurry Up/Busy is selected.
        if "Hurry Up/Busy" in sel:
            for i in class_map.get("Person", []):
                if i >= 0:
                    ids.append(i)
                    rev[i] = "Person"

        ids = list(set(ids)) or None

        # Counting axis:
        #   x = vertical split line: count objects in LEFT/RIGHT regions.
        #   y = horizontal split line: count objects in TOP/BOTTOM regions.
        count_axis = self.config.get("count_axis", "x")
        if count_axis not in ("x", "y"):
            count_axis = "x"

        # Counting line position as a fraction of frame size.
        # For vertical line: 0.30 = Left 30% / Right 70%,
        #                    0.50 = Left 50% / Right 50%,
        #                    0.70 = Left 70% / Right 30%.
        try:
            count_frac = float(self.config.get("count_frac", LINE_FRAC))
        except (TypeError, ValueError):
            count_frac = LINE_FRAC
        if not 0.05 <= count_frac <= 0.95:
            count_frac = LINE_FRAC

        # Vùng nào được xem là IN.
        # Với vertical line:  left  -> before, right  -> after
        # Với horizontal line: top   -> before, bottom -> after
        count_in_side_ui = self.config.get("count_in_side", "right")
        if count_axis == "x":
            count_in_side = "before" if count_in_side_ui == "left" else "after"
        else:
            count_in_side = "before" if count_in_side_ui == "top" else "after"

        # Tính yolo_conf: nếu Luggage được chọn, dùng conf thấp hơn
        # để YOLO không bỏ sót balo/túi xách, lọc lại sau trong _annotate
        yolo_conf = conf
        for c in sel:
            override = CLASS_CONF.get(c)
            if override is not None:
                yolo_conf = min(yolo_conf, override)

        # ── Warmup webcam (cần vài frame để camera ổn định) ──
        src = self.config.get("source", "webcam")
        if src == "webcam":
            for _ in range(5):
                with self._cap_lock:
                    if self.cap is None:
                        break
                    self.cap.read()

        fidx = fps_n = fail_n = 0
        fps_t = time.time()

        while self.running and not self._stop_event.is_set():
            with self._cap_lock:
                cap = self.cap
                if cap is None:
                    break
                ret, frame = cap.read()
            if not ret:
                if src != "webcam":
                    # Video file hết → loop lại từ đầu, nhưng không làm Flask/browser bị ngắt.
                    with self._cap_lock:
                        if self.cap is not None:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                # Webcam: thử lại, không dừng ngay
                fail_n += 1
                if fail_n > 30:      # quá 30 lần liên tiếp mới dừng mềm
                    print("[detector] webcam paused/lost")
                    break
                time.sleep(0.05)
                continue
            fail_n = 0               # reset khi đọc thành công
            fidx += 1; self.stats["frame_no"] = fidx
            if fidx % skip != 0: continue

            h, w = frame.shape[:2]
            if w > 1280:
                frame = cv2.resize(frame, (1280, int(h*1280/w)))
                h, w = frame.shape[:2]
            line_pos = int((w if count_axis == "x" else h) * count_frac)
            now      = time.time()
            cur      = defaultdict(int)
            zone_in  = defaultdict(int)   # current objects in the side configured as IN
            zone_out = defaultdict(int)   # current objects in the side configured as OUT

            try:
                if track:
                    res = self.model.track(frame, conf=yolo_conf, iou=iou,
                        classes=ids, persist=True, verbose=False)
                else:
                    res = self.model(frame, conf=yolo_conf, iou=iou,
                        classes=ids, verbose=False)
                frame, cur, zone_in, zone_out = self._annotate(
                    frame, res, rev, mode, line_pos, count_axis, count_in_side, sel, now)
            except Exception as e:
                print(f"[detector] {e}")

            frame = self._hud(frame, cur, mode, line_pos, count_axis, count_frac, count_in_side, w, h)

            fps_n += 1
            if now - fps_t >= 1.0:
                self.stats["fps"] = round(fps_n/(now-fps_t), 1)
                self.stats["fps_hist"].append(self.stats["fps"])
                fps_n = 0; fps_t = now

            self.stats["current"] = dict(cur)
            self.stats["current_in_zone"] = dict(zone_in)
            self.stats["current_out_zone"] = dict(zone_out)
            self.stats["busy_count"] = int(cur.get("Hurry Up/Busy", 0))
            if fidx % 15 == 0:
                self.stats["timeline"].append({
                    "t": round(now,1), "cur": dict(cur),
                    "in":  dict(self.stats["in_counts"]),
                    "out": dict(self.stats["out_counts"]),
                })

            with self._lock:
                self._frame = frame

        self.running = self.stats["running"] = False
        self._stop_event.set()

    # ── annotate ──────────────────────────────────────────
    def _annotate(self, frame, results, rev, mode, line_pos, count_axis, count_in_side, sel, now):
        cur = defaultdict(int)
        zone_in = defaultdict(int)
        zone_out = defaultdict(int)
        if not results or results[0].boxes is None:
            return frame, cur, zone_in, zone_out
        boxes = results[0].boxes
        has_id = boxes.id is not None
        counting = (mode in ("counting", "detection_counting"))
        fallback_used = set()

        for i, xyxy in enumerate(boxes.xyxy):
            cid = int(boxes.cls[i])
            cf  = float(boxes.conf[i])
            uc  = rev.get(cid)
            if uc is None:
                continue
            # Allow any checkpoint-specific Person ID to be processed as the
            # base class for derived Hurry Up/Busy. This works for both COCO
            # and Open Images V7 / YOLO26 class indices.
            allow_for_hurry = (uc == "Person" and "Hurry Up/Busy" in sel)
            if uc not in sel and not allow_for_hurry:
                continue

            # ── Lọc per-class confidence ──────────────────
            # Person/Pet dùng conf từ UI; Luggage dùng CLASS_CONF override
            cls_min_conf = CLASS_CONF.get(uc)
            if cls_min_conf is None:
                # class này dùng conf từ UI (đã được YOLO filter rồi)
                pass
            elif cf < cls_min_conf:
                continue   # bỏ qua nếu dưới ngưỡng override

            x1,y1,x2,y2 = map(int, xyxy)
            cx,cy = (x1+x2)//2, (y1+y2)//2
            tid = int(boxes.id[i]) if has_id else None

            # Một số trường hợp YOLO track không trả boxes.id.
            # Fallback centroid tracker giúp Total IN/OUT vẫn chạy được.
            if counting and tid is None:
                tid = self._fallback_id(uc, cx, cy, now, fallback_used)

            is_hurry = False
            if tid is not None and "Hurry Up/Busy" in sel and uc == "Person":
                buf = self._vel_buf.setdefault(tid, [])
                buf.append((cx, cy, now))
                if len(buf) > 12:
                    self._vel_buf[tid] = buf[-12:]
                if self._vel(buf) > HURRY_VEL:
                    is_hurry = True

            # If only Hurry Up/Busy is selected, do not display/count normal Person objects
            # until their movement speed passes the Hurry threshold.
            if uc == "Person" and "Person" not in sel and "Hurry Up/Busy" in sel and not is_hurry:
                continue

            disp  = "Hurry Up/Busy" if is_hurry else uc

            if counting:
                # Hybrid logic:
                # 1) RegionCounter-like: count current objects on the IN/OUT side for live occupancy.
                # 2) ObjectCounter-like: accumulate Total IN/OUT only when a tracked object crosses the line.
                pos = cx if count_axis == "x" else cy
                obj_side = "before" if pos < line_pos else "after"
                if obj_side == count_in_side:
                    zone_in[disp] += 1
                else:
                    zone_out[disp] += 1

                # Crossing counter: cumulative Total IN/OUT.
                # This fixes the previous issue where totals were overwritten by current zone counts.
                if tid is not None:
                    event = self._crossing.update(tid, pos, line_pos, now, count_in_side)
                    if event == "IN":
                        self.stats["in_counts"][disp] += 1
                    elif event == "OUT":
                        self.stats["out_counts"][disp] += 1

            color = CLASS_COLOR.get(disp,(128,128,128))
            cur[disp] += 1

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            id_s = f"#{tid} " if tid is not None and mode in ("counting", "detection_counting") else ""

            # ── Hiển thị tên chi tiết cho Luggage / Open Images ─────────
            if uc == "Luggage":
                sub_name = self._class_name(cid)
                if sub_name.isdigit() and cid in LUGGAGE_NAMES:
                    sub_name = LUGGAGE_NAMES[cid]
            elif uc == "Pet":
                sub_name = self._class_name(cid)
                if sub_name.isdigit():
                    sub_name = disp
            elif uc == "Person" and self._norm_name(self._class_name(cid)) != "person":
                sub_name = f"Person · {self._class_name(cid)}"
            else:
                sub_name = disp
            lbl  = f"{id_s}{sub_name} {cf:.2f}"

            lw   = len(lbl)*8+8
            cv2.rectangle(frame,(x1,y1-22),(x1+lw,y1),color,-1)
            cv2.putText(frame,lbl,(x1+3,y1-6),
                cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1,cv2.LINE_AA)
        return frame, cur, zone_in, zone_out

    # ── fallback centroid tracker ──────────────────────────
    def _fallback_id(self, cls_name, cx, cy, now, used):
        """
        Gán ID tạm bằng centroid khi YOLO tracker không trả ID.
        Dùng để Total IN / Total OUT vẫn hoạt động trong các video/phiên bản
        Ultralytics không sinh boxes.id.
        """
        tracks = self._fallback_tracks[cls_name]

        # Xóa track quá cũ
        for tid in list(tracks.keys()):
            if now - tracks[tid]["last"] > 2.0:
                del tracks[tid]

        best_tid, best_d = None, 1e9
        for tid, info in tracks.items():
            if tid in used:
                continue
            d = ((cx - info["cx"]) ** 2 + (cy - info["cy"]) ** 2) ** 0.5
            if d < best_d:
                best_tid, best_d = tid, d

        # Ngưỡng rộng hơn một chút vì frame_skip có thể làm object nhảy xa giữa 2 lần xử lý
        if best_tid is None or best_d > 140:
            best_tid = f"fb{self._next_fallback_id}"
            self._next_fallback_id += 1

        tracks[best_tid] = {"cx": cx, "cy": cy, "last": now}
        used.add(best_tid)
        return best_tid

    # ── HUD ───────────────────────────────────────────────
    def _hud(self, frame, cur, mode, line_pos, count_axis, count_frac, count_in_side, w, h):
        if mode in ("counting", "detection_counting"):
            ov2 = frame.copy()
            if count_axis == "x":
                # Vertical counting line: suitable for people moving left/right.
                cv2.rectangle(ov2,
                              (line_pos - LINE_MARGIN, 0),
                              (line_pos + LINE_MARGIN, h),
                              (200, 200, 0), -1)
                cv2.addWeighted(ov2, 0.10, frame, 0.90, 0, frame)
                cv2.line(frame, (line_pos, 0), (line_pos, h), (0, 60, 230), 2)
                cy = h // 2
                left_pct = int(round(count_frac * 100))
                right_pct = 100 - left_pct
                left_label  = "IN" if count_in_side == "before" else "OUT"
                right_label = "IN" if count_in_side == "after" else "OUT"
                left_color  = (0, 200, 80) if left_label == "IN" else (60, 60, 230)
                right_color = (0, 200, 80) if right_label == "IN" else (60, 60, 230)
                cv2.putText(frame, f"LEFT {left_pct}% = {left_label}", (12, cy - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, left_color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"RIGHT {right_pct}% = {right_label}", (line_pos + 10, cy - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, right_color, 2, cv2.LINE_AA)
                cv2.putText(frame, "CROSSING COUNT: total changes only when object crosses line", (max(8, min(line_pos + 10, w - 560)), cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                # Horizontal counting line: suitable for people moving up/down.
                cv2.rectangle(ov2,
                              (0, line_pos - LINE_MARGIN),
                              (w, line_pos + LINE_MARGIN),
                              (200, 200, 0), -1)
                cv2.addWeighted(ov2, 0.10, frame, 0.90, 0, frame)
                cv2.line(frame, (0, line_pos), (w, line_pos), (0, 60, 230), 2)
                cx = w // 2
                top_pct = int(round(count_frac * 100))
                bottom_pct = 100 - top_pct
                top_label = "IN" if count_in_side == "before" else "OUT"
                bottom_label = "IN" if count_in_side == "after" else "OUT"
                cv2.putText(frame, f"TOP {top_pct}% = {top_label} / BOTTOM {bottom_pct}% = {bottom_label}", (cx - 190, max(24, line_pos - 32)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        items = list(cur.items())
        ph = 28+len(items)*22+8
        ov = frame.copy()
        cv2.rectangle(ov,(6,6),(220,ph),(15,15,15),-1)
        cv2.addWeighted(ov,0.55,frame,0.45,0,frame)
        mlbl = {"detection":"DETECTION MODE","counting":"COUNTING MODE",
                "detection_counting":"DETECTION & COUNTING"}.get(mode,"DETECTION MODE")
        cv2.putText(frame,mlbl,(w-200,22),
            cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,200,80),1,cv2.LINE_AA)
        for i,(cls,cnt) in enumerate(items):
            cv2.putText(frame,f"{cls}: {cnt}",(12,24+i*22),
                cv2.FONT_HERSHEY_SIMPLEX,0.50,
                CLASS_COLOR.get(cls,(200,200,200)),1,cv2.LINE_AA)
        cv2.putText(frame,f"Frame: {self.stats['frame_no']} | FPS: {self.stats['fps']}",
            (8,h-8),cv2.FONT_HERSHEY_SIMPLEX,0.42,(160,160,160),1)
        return frame

    # ── MJPEG ─────────────────────────────────────────────
    def generate_frames(self):
        """
        Keep the MJPEG endpoint alive even when detection is stopped.
        Browser tab switches or Stop should not kill the Flask app; the stream
        simply keeps serving the last frame / placeholder until Start is pressed again.
        """
        ph = self._ph()
        try:
            while True:
                with self._lock:
                    f = self._frame
                if f is None:
                    f = ph
                ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + buf.tobytes() + b"\r\n")
                time.sleep(0.033)
        except GeneratorExit:
            return
        except Exception as e:
            print(f"[video_feed] stream paused: {e}")
            return

    @staticmethod
    def _ph():
        img = np.zeros((480,640,3),np.uint8)
        cv2.putText(img,"Configure & press Start Detection",
            (70,245),cv2.FONT_HERSHEY_SIMPLEX,0.7,(80,80,80),1)
        return img

    # ── stats / sessions ──────────────────────────────────
    def get_stats(self):
        s  = self.stats
        ti = dict(s["in_counts"]); to = dict(s["out_counts"])
        si = sum(ti.values());     so = sum(to.values())
        return {
            "running":   s["running"],  "fps": s["fps"],
            "frame_no":  s["frame_no"], "current": dict(s["current"]),
            "in_counts": ti, "out_counts": to,
            "total_in":  si, "total_out": so, "net": si-so,
            "avg_fps":   round(sum(s["fps_hist"])/len(s["fps_hist"]),1)
                         if s["fps_hist"] else 0.0,
            "current_in_zone": dict(s.get("current_in_zone", {})),
            "current_out_zone": dict(s.get("current_out_zone", {})),
            "busy_count": int(s.get("busy_count", 0)),
            "timeline":  list(s["timeline"])[-60:],
            "model":     self.model_name,
            "mode":      self.config.get("mode","detection"),
        }

    def get_sessions(self):
        p = os.path.join("logs","sessions.json")
        if not os.path.exists(p): return []
        try: return json.load(open(p))
        except: return []

    def _save_session(self):
        s = self.stats
        if not s.get("start_time"): return
        os.makedirs("logs", exist_ok=True)
        p = os.path.join("logs","sessions.json")
        db = []
        if os.path.exists(p):
            try: db = json.load(open(p))
            except: pass
        ti = dict(s["in_counts"]); to = dict(s["out_counts"])
        si = sum(ti.values());     so = sum(to.values())
        db.append({
            "timestamp": datetime.now().isoformat(),
            "model":     self.model_name or "—",
            "mode":      self.config.get("mode","detection"),
            "source":    self.config.get("source","—"),
            "in_counts": ti, "out_counts": to,
            "total_in":  si, "total_out":  so, "net": si-so,
            "avg_fps":   round(sum(s["fps_hist"])/len(s["fps_hist"]),1)
                         if s["fps_hist"] else 0.0,
            "duration_s": round(time.time()-time.mktime(
                datetime.fromisoformat(s["start_time"]).timetuple()),1),
        })
        json.dump(db, open(p,"w"), indent=2)

    @staticmethod
    def _vel(pts):
        if len(pts)<2: return 0.0
        x0,y0,t0=pts[0]; xn,yn,tn=pts[-1]
        dt=tn-t0
        return 0.0 if dt<=0 else np.sqrt((xn-x0)**2+(yn-y0)**2)/dt