import cv2
import time
import threading
import numpy as np
from collections import defaultdict
from .models import ClassroomEvent

# ---------------- CONFIG ----------------
MODEL_NAME          = "yolov8s.pt"
INFER_SIZE          = 416
CONF_THRESHOLD      = 0.35
IOU_THRESHOLD       = 0.45
EXIT_TIME           = 4.0
ALARM_FLAG_DURATION = 4.0

MIN_WIDTH  = 30
MIN_HEIGHT = 60
MIN_RATIO  = 0.15
MAX_RATIO  = 1.8


# ---------------- CAMERA THREAD ----------------
class _CameraReader(threading.Thread):
    def __init__(self, camera_index: int):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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


# ---------------- IMAGE ENHANCE ----------------
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def _enhance(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ---------------- ENGINE ----------------
class ClassroomEngine:

    def __init__(self, camera_index: int = 0):
        from ultralytics import YOLO

        self.model = YOLO(MODEL_NAME)

        # warm-up
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)

        self.camera_index = camera_index
        self._init_state()

    # ---------------- STATE ----------------
    def _init_state(self):
        self.last_seen     = {}
        self.present_ids   = set()
        self.present_count = 0
        self.history       = []
        self.alarm_until   = 0.0

        self.db_records    = {}

        self._output_frame = None
        self._output_lock  = threading.Lock()

        # NEW FIX
        self.started = False
        self._frame_ready_count = 0

    # ---------------- INFERENCE ----------------
    def _run_inference(self, frame):
        enhanced = _enhance(frame)

        results = self.model.track(
            enhanced,
            persist=True,
            classes=[0],
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=INFER_SIZE,
            verbose=False,
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

    # ---------------- STATE UPDATE ----------------
    def _update_state(self, detections):
        # BLOCK EARLY EXECUTION
        if not self.started:
            return

        current_time = time.time()
        current_ids = set()

        for tid, *_ in detections:
            current_ids.add(tid)
            self.last_seen[tid] = current_time

            if tid not in self.present_ids:
                self.present_ids.add(tid)
                self.present_count += 1

                self.history.append({
                    "time": time.strftime("%H:%M:%S"),
                    "id": tid,
                    "event": "ENTERED"
                })

                event_obj = ClassroomEvent.objects.create(event="ENTERED")
                self.db_records[tid] = event_obj

        for tid in list(self.present_ids):
            if tid not in current_ids:
                elapsed = current_time - self.last_seen.get(tid, 0)

                if elapsed > EXIT_TIME:
                    self.present_ids.discard(tid)
                    self.present_count = max(0, self.present_count - 1)

                    self.history.append({
                        "time": time.strftime("%H:%M:%S"),
                        "id": tid,
                        "event": "LEFT"
                    })

                    event_obj = self.db_records.get(tid)
                    if event_obj:
                        event_obj.event = "LEFT"
                        event_obj.save()
                        self.db_records.pop(tid, None)
                    else:
                        ClassroomEvent.objects.create(event="LEFT")

                    self.last_seen.pop(tid, None)
                    self.alarm_until = current_time + ALARM_FLAG_DURATION

    # ---------------- ANNOTATE ----------------
    def _annotate(self, frame, detections):
        out = frame.copy()
        h, w = out.shape[:2]

        alarm_active = time.time() < self.alarm_until

        for tid, x1, y1, x2, y2, conf in detections:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 80), 2)
            cv2.putText(out, f"ID {tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if alarm_active:
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)

        return out

    # ---------------- WORKER ----------------
    def _inference_worker(self, reader, stop_event):
        while not stop_event.is_set():
            frame = reader.read()
            if frame is None:
                time.sleep(0.01)
                continue

            detections = self._run_inference(frame)

            annotated = self._annotate(frame, detections)

            # enable after few frames
            if not self.started:
                self._frame_ready_count += 1
                if self._frame_ready_count > 5:
                    self.started = True

            self._update_state(detections)

            with self._output_lock:
                self._output_frame = annotated

    # ---------------- STREAM ----------------
    def generate_frames(self):
        reader = _CameraReader(self.camera_index)
        stop_ev = threading.Event()

        worker = threading.Thread(
            target=self._inference_worker,
            args=(reader, stop_ev),
            daemon=True
        )

        reader.start()
        worker.start()

        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Loading AI Model...", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        try:
            while True:
                with self._output_lock:
                    frame = self._output_frame

                display = placeholder if frame is None else frame

                _, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 82])

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')

                time.sleep(0.033)

        finally:
            stop_ev.set()
            reader.stop()

    # ---------------- STATS ----------------
    def get_stats(self):
        from .shared_state import shared_state

        total_students = 10 # default students count
        missing = total_students - shared_state.face_present_count

        return {
            "total_students": total_students,
            "present_count": self.present_count,
            "present_ids": list(self.present_ids),
            "history": self.history[-50:],
            "alarm": time.time() < self.alarm_until,
            "face_present_count": shared_state.face_present_count,
            "difference": max(0, missing),
        }

    def reset(self):
        self._init_state()