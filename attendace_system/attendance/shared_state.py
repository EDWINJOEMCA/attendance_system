# attendance/shared_state.py

class AttendanceState:
    def __init__(self):
        # From Face Engine
        self.face_present_count = 0
        self.face_log = []

shared_state = AttendanceState()