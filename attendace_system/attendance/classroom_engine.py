"""
classroom_engine.py  — v3  (threaded, non-blocking)
─────────────────────────────────────────────────────────────────
Architecture
────────────
  Thread A  — CameraReader   : cap.read() at full camera FPS,
                               always keeps the LATEST frame ready.
  Thread B  — InferenceWorker: pulls latest frame, runs YOLOv8s,
                               pushes annotated result + detections.
  Main loop — generate_frames: reads the latest annotated frame and
                               yields MJPEG as fast as encoding allows.

This decouples the slow YOLO inference (Thread B) from both the
camera capture (Thread A) and the stream output (main loop),
so the browser feed is always smooth and counts are always current.

Key choices
───────────
  • YOLOv8s   — small model: ~2-3× faster than medium, still much
                better than nano because we use imgsz=416 (smaller
                inference resolution halves latency again).
  • No flip augmentation — removed; it was doubling inference time
                and the tracker handles edge-cases better than we do.
  • CLAHE kept — cheap CPU op, meaningfully improves dim-room recall.
  • Inference every frame — no skip needed because the worker thread
                naturally rate-limits itself to its own throughput.
  • EXIT_TIME = 4 s — longer grace period so brief occlusions don't
                fire false "left" alarms.
"""

import cv2
import time
import threading
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_NAME          = "yolov8s.pt"   # small: fast + accurate enough for classrooms
INFER_SIZE          = 416            # inference resolution (480 → 416 = ~30% faster)
CONF_THRESHOLD      = 0.35
IOU_THRESHOLD       = 0.45
EXIT_TIME           = 4.0            # seconds absent → student "left"
ALARM_FLAG_DURATION = 4.0            # seconds alarm flag stays true

# Relaxed size guards (seated / partially-visible students)
MIN_WIDTH  = 30
MIN_HEIGHT = 60
MIN_RATIO  = 0.15
MAX_RATIO  = 1.8


# ─────────────────────────────────────────────────────────────
#  Thread A — camera reader (always-fresh frame, non-blocking)
# ─────────────────────────────────────────────────────────────
class _CameraReader(threading.Thread):
    def __init__(self, camera_index: int):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # keep only newest frame
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        self.frame = None
        self.lock  = threading.Lock()
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self._stop.set()
        self.cap.release()


# ─────────────────────────────────────────────────────────────
#  CLAHE enhancer (CPU, cheap)
# ─────────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def _enhance(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────
#  Main engine
# ─────────────────────────────────────────────────────────────
class ClassroomEngine:

    def __init__(self, camera_index: int = 0):
        from ultralytics import YOLO
        self.model        = YOLO(MODEL_NAME)
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)
        self.camera_index = camera_index
        self._init_state()

    # ── state ────────────────────────────────────────────────
    def _init_state(self):
        self.last_seen     : dict[int, float] = {}
        self.present_ids   : set[int]         = set()
        self.present_count : int              = 0
        self.history       : list[dict]       = []
        self.alarm_until   : float            = 0.0
        # latest annotated frame shared between inference worker and MJPEG loop
        self._output_frame : np.ndarray | None = None
        self._output_lock  = threading.Lock()

    # ── CLAHE + YOLO inference (runs in worker thread) ───────
    def _run_inference(self, frame) -> list[tuple]:
        """Returns list of (tid, x1, y1, x2, y2, conf) for valid detections."""
        enhanced = _enhance(frame)
        results  = self.model.track(
            enhanced,
            persist     = True,
            classes     = [0],
            conf        = CONF_THRESHOLD,
            iou         = IOU_THRESHOLD,
            imgsz       = INFER_SIZE,
            verbose     = False,
        )
        detections = []
        if results[0].boxes.id is None:
            return detections
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids   = results[0].boxes.id.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        for box, tid, conf in zip(boxes, ids, confs):
            if conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box)
            bw, bh = x2 - x1, y2 - y1
            if bw < MIN_WIDTH or bh < MIN_HEIGHT:
                continue
            ratio = bw / max(bh, 1)
            if ratio < MIN_RATIO or ratio > MAX_RATIO:
                continue
            detections.append((int(tid), x1, y1, x2, y2, float(conf)))
        return detections

    # ── state update (called in worker thread) ───────────────
    def _update_state(self, detections: list[tuple]):
        current_time = time.time()
        current_ids: set[int] = set()

        for tid, *_ in detections:
            current_ids.add(tid)
            self.last_seen[tid] = current_time
            if tid not in self.present_ids:
                self.present_ids.add(tid)
                self.present_count += 1
                self.history.append({
                    "time": time.strftime("%H:%M:%S"),
                    "id": tid, "event": "ENTERED"
                })

        for tid in list(self.present_ids):
            if tid not in current_ids:
                elapsed = current_time - self.last_seen.get(tid, 0)
                if elapsed > EXIT_TIME:
                    self.present_ids.discard(tid)
                    self.present_count = max(0, self.present_count - 1)
                    self.history.append({
                        "time": time.strftime("%H:%M:%S"),
                        "id": tid, "event": "LEFT"
                    })
                    self.last_seen.pop(tid, None)
                    self.alarm_until = current_time + ALARM_FLAG_DURATION

    # ── draw annotations onto a frame copy ───────────────────
    def _annotate(self, frame, detections: list[tuple]) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        alarm_active = time.time() < self.alarm_until

        # Per-person boxes
        for tid, x1, y1, x2, y2, conf in detections:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 80), 2)
            label = f"ID {tid}  {conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(out, (x1, y1 - lh - 10), (x1 + lw + 6, y1), (0, 255, 80), -1)
            cv2.putText(out, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # Alarm overlay
        if alarm_active:
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)

        # Top bar
        bar_color = (40, 0, 0) if alarm_active else (10, 10, 10)
        cv2.rectangle(out, (0, 0), (w, 58), bar_color, -1)
        cv2.putText(out, "CLASSROOM COUNTER", (12, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 255), 1)
        cv2.putText(out, f"Present: {self.present_count}", (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 130), 2)
        if alarm_active:
            alarm_text = "! STUDENT LEFT"
            (aw, _), _ = cv2.getTextSize(alarm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.putText(out, alarm_text, (w - aw - 15, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 80, 255), 2)

        # Bottom bar
        cv2.rectangle(out, (0, h - 30), (w, h), (10, 10, 10), -1)
        cv2.putText(out, "Tracking students in classroom...", (12, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        return out

    # ── Thread B — inference worker ───────────────────────────
    def _inference_worker(self, reader: _CameraReader, stop_event: threading.Event):
        while not stop_event.is_set():
            frame = reader.read()
            if frame is None:
                time.sleep(0.01)
                continue

            detections = self._run_inference(frame)
            self._update_state(detections)
            annotated  = self._annotate(frame, detections)

            with self._output_lock:
                self._output_frame = annotated

    # ── MJPEG stream generator ────────────────────────────────
    def generate_frames(self):
        reader    = _CameraReader(self.camera_index)
        stop_ev   = threading.Event()
        worker    = threading.Thread(
            target=self._inference_worker,
            args=(reader, stop_ev),
            daemon=True,
        )
        reader.start()
        worker.start()

        # Blank placeholder shown before first inference finishes
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Loading AI Model...", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        try:
            while True:
                with self._output_lock:
                    frame = self._output_frame

                display = placeholder if frame is None else frame
                _, buf  = cv2.imencode('.jpg', display,
                                       [cv2.IMWRITE_JPEG_QUALITY, 82])
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + buf.tobytes() + b'\r\n'
                )
                # Yield at ~30 fps max regardless of inference speed
                time.sleep(0.033)
        finally:
            stop_ev.set()
            reader.stop()

    # ─────────────────────────────────────────────────────────
    def get_stats(self):

        from .shared_state import shared_state   # NEW IMPORT

        missing = shared_state.face_present_count - self.present_count
        extra = self.present_count - shared_state.face_present_count

        return {
            "present_count": self.present_count,
            "present_ids":   list(self.present_ids),
            "history":       self.history[-50:],
            "alarm":         time.time() < self.alarm_until,
            # NEW DATA FROM FACE ENGINE
            "face_present_count": shared_state.face_present_count,
            "difference": max(0, missing),
            "extra": max(0, extra)
        }

    def reset(self):
        self._init_state()
