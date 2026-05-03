"""
face_engine.py
─────────────────────────────────────────────────────────────────
Face-gesture attendance engine.

  - Student stands still → RECOGNISED
  - Move RIGHT → PRESENT (count++)
  - Move LEFT  → CANCELLED
  - Must fully clear the frame before next student is processed.

Streams MJPEG via generate_frames() for Django's StreamingHttpResponse.
"""

import cv2
import time
import urllib.request
import os
from collections import deque
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from .models import FaceAttendance

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")

STABLE_FRAMES_REQUIRED    = 20
DIRECTION_FRAMES_REQUIRED = 15
EMPTY_FRAMES_REQUIRED     = 20
MOVEMENT_THRESHOLD        = 0.045
MIN_FACE_CONFIDENCE       = 0.65


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[FaceEngine] Downloading MediaPipe model → {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[FaceEngine] Download complete.")


# ─────────────────────────────────────────────────────────────
#  STATES
# ─────────────────────────────────────────────────────────────
class State:
    WAITING_FOR_EXIT = "WAITING_FOR_EXIT"
    WAITING          = "WAITING"
    DETECTING        = "DETECTING"
    RECOGNISED       = "RECOGNISED"
    DONE             = "DONE"


# ─────────────────────────────────────────────────────────────
#  ENGINE
# ─────────────────────────────────────────────────────────────
class FaceAttendanceEngine:

    def __init__(self, camera_index: int = 0):
        ensure_model()

        base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        opts = mp_vision.FaceDetectorOptions(
            base_options=base_opts,
            min_detection_confidence=MIN_FACE_CONFIDENCE,
            min_suppression_threshold=0.3,
        )
        self.detector = mp_vision.FaceDetector.create_from_options(opts)
        self.camera_index = camera_index
        self._reset_state()

    # ── public state ────────────────────────────────────────
    def _reset_state(self):
        self.state            = State.WAITING_FOR_EXIT
        self.count            = 0
        self.face_cx_history  = deque(maxlen=40)
        self.direction_votes  = deque(maxlen=DIRECTION_FRAMES_REQUIRED)
        self.entry_cx         = None
        self.empty_frame_count = 0
        self.last_result      = None
        self.result_until     = 0
        self.status_msg       = "Please clear the frame..."
        self.status_color     = (80, 80, 80)
        self.log: list[dict]  = []

    # ── core update ─────────────────────────────────────────
    def update(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)
        valid  = [
            d for d in result.detections
            if d.categories and d.categories[0].score >= MIN_FACE_CONFIDENCE
        ]
        face_present = len(valid) > 0

        if self.state == State.WAITING_FOR_EXIT:
            if not face_present:
                self.empty_frame_count += 1
                pct = min(self.empty_frame_count / EMPTY_FRAMES_REQUIRED, 1.0)
                self.status_msg   = f"Clearing... ({int(pct*100)}%)"
                self.status_color = (80, 80, 80)
                if self.empty_frame_count >= EMPTY_FRAMES_REQUIRED:
                    self._enter_waiting()
            else:
                self.empty_frame_count = 0
                self.status_msg   = "Please exit the frame fully..."
                self.status_color = (0, 80, 200)
            return valid

        if self.state == State.WAITING:
            if face_present:
                best = self._best(valid)
                cx   = self._cx(best, w)
                self.face_cx_history.clear()
                self.face_cx_history.append(cx)
                self.state        = State.DETECTING
                self.status_msg   = "Face detected — hold still..."
                self.status_color = (255, 200, 0)
            return valid

        if self.state == State.DETECTING:
            if not face_present:
                self.face_cx_history.clear()
                self.state        = State.WAITING
                self.status_msg   = "Waiting for student..."
                self.status_color = (200, 200, 200)
                return valid
            cx = self._cx(self._best(valid), w)
            self.face_cx_history.append(cx)
            if len(self.face_cx_history) >= STABLE_FRAMES_REQUIRED:
                spread = max(self.face_cx_history) - min(self.face_cx_history)
                if spread < MOVEMENT_THRESHOLD:
                    self.entry_cx = cx
                    self.direction_votes.clear()
                    self.state        = State.RECOGNISED
                    self.status_msg   = "✓ Recognised!  →RIGHT=Present  LEFT←=Cancel"
                    self.status_color = (0, 255, 150)
                else:
                    self.face_cx_history.popleft()
            return valid

        if self.state == State.RECOGNISED:
            if not face_present:
                self._finish("ignored", (100, 100, 0), "Walked away — not counted")
                return valid
            cx    = self._cx(self._best(valid), w)
            delta = cx - self.entry_cx
            if abs(delta) > MOVEMENT_THRESHOLD / 2:
                self.direction_votes.append("right" if delta > 0 else "left")
            if len(self.direction_votes) >= DIRECTION_FRAMES_REQUIRED:
                rights = self.direction_votes.count("right")
                lefts  = self.direction_votes.count("left")
                if rights > lefts:
                    self._mark_present()
                elif lefts > rights:
                    self._mark_cancelled()
            return valid

        return valid  # DONE state — result shown until exit

    # ── helpers ─────────────────────────────────────────────
    def _best(self, detections):
        return max(detections, key=lambda d: d.categories[0].score)

    def _cx(self, det, w):
        bb = det.bounding_box
        return (bb.origin_x + bb.width / 2) / w

    def _enter_waiting(self):
        self.state             = State.WAITING
        self.empty_frame_count = 0
        self.face_cx_history.clear()
        self.direction_votes.clear()
        self.entry_cx          = None
        self.status_msg        = "Waiting for student..."
        self.status_color      = (200, 200, 200)

    def _mark_present(self):
        from .shared_state import shared_state
        
        self.count += 1
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append({"num": self.count, "time": ts, "status": "PRESENT"})
        FaceAttendance.objects.create(status="PRESENT") # save on DB
        
        # UPDATE SHARED STATE
        shared_state.face_present_count = self.count
        shared_state.face_log = self.log

        self._finish("present", (0, 230, 80), f" PRESENT — Total: {self.count}")

    def _mark_cancelled(self):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append({"num": None, "time": ts, "status": "CANCELLED"})
        FaceAttendance.objects.create(status="CANCELLED")  # save on DB
        self._finish("cancelled", (0, 60, 220), " CANCELLED")

    def _finish(self, result, color, msg):
        self.last_result      = result
        self.result_until     = time.time() + 2.0
        self.status_msg       = msg
        self.status_color     = color
        self.empty_frame_count = 0
        self.state            = State.WAITING_FOR_EXIT

    # ── draw ────────────────────────────────────────────────
    def draw(self, frame, detections):
        h, w = frame.shape[:2]

        # Result flash overlay
        if time.time() < self.result_until and self.last_result:
            overlay = frame.copy()
            color   = (0, 200, 60) if self.last_result == "present" else (0, 50, 200)
            cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        box_color = {
            State.WAITING_FOR_EXIT: (60, 60, 60),
            State.WAITING:          (160, 160, 160),
            State.DETECTING:        (0, 210, 255),
            State.RECOGNISED:       (0, 255, 140),
            State.DONE:             (120, 120, 120),
        }.get(self.state, (180, 180, 180))

        for det in detections:
            bb     = det.bounding_box
            x1, y1 = int(bb.origin_x), int(bb.origin_y)
            x2, y2 = int(x1 + bb.width), int(y1 + bb.height)
            score  = det.categories[0].score if det.categories else 0
            c, t   = 24, 3
            for px, py in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                sx = 1 if px == x1 else -1
                sy = 1 if py == y1 else -1
                cv2.line(frame,(px,py),(px+sx*c,py), box_color, t)
                cv2.line(frame,(px,py),(px,py+sy*c), box_color, t)
            cv2.putText(frame, f"{score:.0%}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Stability bar
        if self.state == State.DETECTING and self.face_cx_history:
            prog = min(len(self.face_cx_history)/STABLE_FRAMES_REQUIRED, 1.0)
            bw = int(w*0.55); bx = (w-bw)//2; by = h-65
            cv2.rectangle(frame,(bx,by),(bx+bw,by+14),(35,35,35),-1)
            cv2.rectangle(frame,(bx,by),(bx+int(bw*prog),by+14),(0,210,255),-1)
            cv2.putText(frame,"Stabilising...",(bx,by-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,210,255),1)

        # Exit gate bar
        if self.state == State.WAITING_FOR_EXIT:
            prog = min(self.empty_frame_count/EMPTY_FRAMES_REQUIRED, 1.0)
            bw = int(w*0.55); bx = (w-bw)//2; by = h-65
            cv2.rectangle(frame,(bx,by),(bx+bw,by+14),(35,35,35),-1)
            cv2.rectangle(frame,(bx,by),(bx+int(bw*prog),by+14),(80,80,80),-1)
            cv2.putText(frame,"Waiting for clear frame...",(bx,by-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(120,120,120),1)

        # Direction arrows
        if self.state == State.RECOGNISED:
            my = h//2
            cv2.arrowedLine(frame,(w//2+40,my),(w//2+140,my),(0,230,80),3,tipLength=0.35)
            cv2.putText(frame,"PRESENT",(w//2+145,my+6),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,230,80),2)
            cv2.arrowedLine(frame,(w//2-40,my),(w//2-140,my),(0,70,220),3,tipLength=0.35)
            cv2.putText(frame,"CANCEL",(w//2-225,my+6),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,70,220),2)

        # Top bar
        cv2.rectangle(frame,(0,0),(w,50),(12,12,12),-1)
        cv2.putText(frame,f"[{self.state}]",(10,33),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,self.status_color,2)
        cnt = f"Present: {self.count}"
        (tw,_),_ = cv2.getTextSize(cnt,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
        cv2.putText(frame,cnt,(w-tw-15,33),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,100),2)

        # Bottom bar
        cv2.rectangle(frame,(0,h-36),(w,h),(12,12,12),-1)
        cv2.putText(frame,self.status_msg,(12,h-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,self.status_color,2)

        return frame

    # ── MJPEG stream ────────────────────────────────────────
    def generate_frames(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame      = cv2.flip(frame, 1)
                detections = self.update(frame)
                frame      = self.draw(frame, detections)
                _, buf     = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
                )
        finally:
            cap.release()

    def close(self):
        self.detector.close()

    def get_stats(self):
        return {
            "count":  self.count,
            "state":  self.state,
            "status": self.status_msg,
            "log":    self.log,
        }
