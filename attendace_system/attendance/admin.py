from django.contrib import admin
from .models import FaceAttendance, ClassroomEvent

admin.site.register(FaceAttendance)
admin.site.register(ClassroomEvent)