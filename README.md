Here’s your **fully updated `README.md`** with all the new features you added:

* Face ↔ Classroom integration
* Missing & Extra count
* Alarm system
* Performance improvements
* Preloading optimization

---

```markdown
# Smart Attendance System — Django + AI

A unified **AI-powered attendance system** combining:

- Face Recognition (individual attendance)
- 📹 Classroom Monitoring (real-time verification)

---

## Features

| Mode | Tech | Description |
|------|------|------------|
| **Face Recognition** | MediaPipe BlazeFace | Student stands → recognised → swipe right = Present |
| **Classroom Counter** | YOLOv8 + Tracking | Detects and tracks all people inside classroom |
| **Attendance Sync** | Shared State | Compares face attendance vs classroom presence |
| **Missing Detection** | Logic Layer | Shows how many students didn’t enter |
| **Extra Detection** | Logic Layer | Detects unauthorized / extra people |
| **Audio Alert** | Web Audio API | Plays alarm when student leaves classroom |

---

## System Workflow

### Step 1 — Face Attendance
- Students mark attendance one by one
- System stores:
  - Total present count
  - Attendance log

### Step 2 — Classroom Monitoring
- YOLO detects all persons in classroom
- Assigns unique tracking IDs
- Tracks entry & exit

### Step 3 — Smart Verification

| Metric | Meaning |
|-------|--------|
| **Present Now** | People currently inside classroom |
| **Face Present** | Students marked present |
| **Missing** | Face Present − Classroom Count |
| **Extra** | Classroom Count − Face Present |

---

## Project Structure

```

attendance_project/
├── manage.py
├── requirements.txt
├── attendance_project/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── attendance/
│   ├── face_engine.py
│   ├── classroom_engine.py
│   ├── shared_state.py      # NEW (connects both modes)
│   ├── views.py
│   └── urls.py
├── templates/attendance/
│   ├── base.html
│   ├── home.html
│   ├── face.html
│   └── classroom.html
└── static/

````

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
````

---

### 2. Run server

```bash
python manage.py runserver
```

Open:

```
http://127.0.0.1:8000
```

---

## Mode 1 — Face Recognition (`/face/`)

1. Open Face Recognition
2. Student stands in front of camera
3. Wait until **RECOGNISED**
4. Action:

   * Move RIGHT → PRESENT
   * Move LEFT → CANCEL

 Count increases
 Stored in shared state

---

## Mode 2 — Classroom Counter (`/classroom/`)

1. Open Classroom page
2. Camera detects all people
3. Tracks with unique IDs
4. Displays:

* Present Now
* Face Present
* Missing
* Extra

---

## Audio Alert System

* Alarm triggers when a student **leaves**
* Uses **Web Audio API**
* Requires **one user click to enable sound**

---

## API Endpoints

| Method | URL                  | Description                 |
| ------ | -------------------- | --------------------------- |
| GET    | `/face/stream/`      | Face camera stream          |
| GET    | `/face/stats/`       | Face attendance data        |
| POST   | `/face/reset/`       | Reset face system           |
| GET    | `/classroom/stream/` | Classroom stream            |
| GET    | `/classroom/stats/`  | Classroom + comparison data |
| POST   | `/classroom/reset/`  | Reset classroom             |

---

## Performance Optimizations (Implemented)

* YOLO model preloaded at server start
* Warm-up inference added
* Reduced frame resolution
* Faster model (yolov8n)
* Frame skipping (optional)
* MJPEG streaming optimized

---

## Limitations

* Classroom detects **all humans**, not specific students
* Missing/Extra is **count-based**, not identity-based
* False positives possible in crowded environments

---

## Notes

* YOLO model downloads automatically
* MediaPipe model downloads automatically
* Works best with:

  * Good lighting
  * Stable camera
  * Clear face visibility

---

## Tech Stack

* Django
* OpenCV
* MediaPipe
* YOLOv8 (Ultralytics)
* JavaScript (Web Audio API)
