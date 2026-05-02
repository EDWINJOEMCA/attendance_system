"""
views.py  —  Smart Attendance System
─────────────────────────────────────────────────────────────────
Two modes, two camera streams, one Django backend.

  GET  /                      → Home / mode selector
  GET  /face/                 → Face-gesture attendance page
  GET  /face/stream/          → MJPEG stream (face mode)
  GET  /face/stats/           → JSON stats (face mode)
  POST /face/reset/           → Reset face engine
  GET  /classroom/            → Classroom counter page
  GET  /classroom/stream/     → MJPEG stream (classroom mode)
  GET  /classroom/stats/      → JSON stats (classroom mode)
  POST /classroom/reset/      → Reset classroom engine
"""

import json
import threading
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

# ── Engine singletons (one per mode) ────────────────────────
_face_engine      = None
_classroom_engine = None
_face_lock        = threading.Lock()
_classroom_lock   = threading.Lock()


def _get_face_engine():
    global _face_engine
    with _face_lock:
        if _face_engine is None:
            from .face_engine import FaceAttendanceEngine
            _face_engine = FaceAttendanceEngine(camera_index=0)
    return _face_engine


def _get_classroom_engine():
    global _classroom_engine
    with _classroom_lock:
        if _classroom_engine is None:
            from .classroom_engine import ClassroomEngine
            _classroom_engine = ClassroomEngine(camera_index=0)
    return _classroom_engine


# ── Home ────────────────────────────────────────────────────
def home(request):
    return render(request, 'attendance/home.html')


# ── Face mode ───────────────────────────────────────────────
def face_page(request):
    return render(request, 'attendance/face.html')


def face_stream(request):
    engine = _get_face_engine()
    response = StreamingHttpResponse(
        engine.generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
    response['Cache-Control'] = 'no-cache'
    return response


def face_stats(request):
    engine = _get_face_engine()
    return JsonResponse(engine.get_stats())


@csrf_exempt
@require_POST
def face_reset(request):
    global _face_engine
    with _face_lock:
        if _face_engine is not None:
            _face_engine.close()
            _face_engine = None
    return JsonResponse({"ok": True})


# ── Classroom mode ───────────────────────────────────────────
def classroom_page(request):
    return render(request, 'attendance/classroom.html')


def classroom_stream(request):
    engine = _get_classroom_engine()
    response = StreamingHttpResponse(
        engine.generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
    response['Cache-Control'] = 'no-cache'
    return response


def classroom_stats(request):
    engine = _get_classroom_engine()
    return JsonResponse(engine.get_stats())


@csrf_exempt
@require_POST
def classroom_reset(request):
    global _classroom_engine
    with _classroom_lock:
        if _classroom_engine is not None:
            _classroom_engine.reset()
    return JsonResponse({"ok": True})


# ── PRELOAD ENGINES AT SERVER START ────────────────────────
def preload_engines():
    global _classroom_engine
    from .classroom_engine import ClassroomEngine

    print("[INFO] Preloading Classroom Engine...")
    _classroom_engine = ClassroomEngine(camera_index=0)
    print("[INFO] Classroom Engine Ready!")

# Call immediately when Django starts
preload_engines()