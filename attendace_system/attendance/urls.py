from django.urls import path
from . import views

urlpatterns = [
    # Home
    path('', views.home, name='home'),

    # ── Face-gesture attendance ──────────────────────────────
    path('face/',          views.face_page,      name='face_page'),
    path('face/stream/',   views.face_stream,    name='face_stream'),
    path('face/stats/',    views.face_stats,     name='face_stats'),
    path('face/reset/',    views.face_reset,     name='face_reset'),

    # ── Classroom counter ────────────────────────────────────
    path('classroom/',         views.classroom_page,    name='classroom_page'),
    path('classroom/stream/',  views.classroom_stream,  name='classroom_stream'),
    path('classroom/stats/',   views.classroom_stats,   name='classroom_stats'),
    path('classroom/reset/',   views.classroom_reset,   name='classroom_reset'),
]
