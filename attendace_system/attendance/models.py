from django.db import models


ATTENDANCE_CHOICES = (
    ("PRESENT", "PRESENT"),
    ("CANCELLED", "CANCELLED")
)

EVENT_CHOICES = (
    ("ENTERED", "ENTERED"), 
    ("LEFT", "LEFT")
)

class FaceAttendance(models.Model):
    datetime = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=ATTENDANCE_CHOICES)


class ClassroomEvent(models.Model):
    datetime = models.DateTimeField(auto_now_add=True)
    event = models.CharField(max_length=20, choices=EVENT_CHOICES)
