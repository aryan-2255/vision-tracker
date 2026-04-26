# Real-Time Object Detection & Tracking

Real-time object detection and multi-object tracking using RF-DETR and ByteTrack.
Works on Mac, Windows, and Linux. Supports webcam, USB camera, and phone camera.

## Features

- Real-time object detection using RF-DETR (Nano / Small / Large)
- Multi-object tracking with ByteTrack (persistent IDs across frames)
- Motion trail visualization
- Phone camera support via IP Webcam (Android) or DroidCam
- Live controls — switch model size, toggle tracker, adjust confidence
- Cross-platform: Mac, Windows, Linux

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create virtual environment
```bash
# Mac / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

For Windows with NVIDIA GPU (recommended):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Run

```bash
python detect_webcam.py
```

First run will download model weights (~130 MB). After that it starts instantly.

## Phone Camera Setup

### Android — IP Webcam App
1. Install **IP Webcam** from Play Store (free)
2. Open app → tap **Start Server**
3. Note the IP shown e.g. `http://192.168.1.5:8080`
4. Edit `detect_webcam.py`:
```python
CAMERA_SOURCE = "http://192.168.1.5:8080/video"
```

### Android / iPhone — DroidCam
1. Install **DroidCam** on phone and PC
2. Connect over WiFi or USB
3. Set `CAMERA_SOURCE = 1` (or 2) in the script

## Controls

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `+` | Increase confidence threshold |
| `-` | Decrease confidence threshold |
| `T` | Toggle ByteTrack tracker on/off |
| `L` | Toggle labels on/off |
| `M` | Cycle model: Nano → Small → Large |

## Model Sizes

| Model | Accuracy (AP) | Speed | Best For |
|-------|-------------|-------|----------|
| Nano  | 48.4 | Fastest | Low-end hardware |
| Small | 53.0 | Fast | Balanced (default) |
| Large | 56.5 | Moderate | Best accuracy |

## Tech Stack

- [RF-DETR](https://github.com/roboflow/rf-detr) — Real-Time Detection Transformer
- [Trackers](https://github.com/roboflow/trackers) — ByteTrack multi-object tracking
- [Supervision](https://github.com/roboflow/supervision) — Annotation utilities
- [OpenCV](https://opencv.org/) — Camera capture and display

## License

This project is licensed under MIT.
RF-DETR (Nano–Large) is Apache 2.0. See [rf-detr license](https://github.com/roboflow/rf-detr/blob/main/LICENSE).
