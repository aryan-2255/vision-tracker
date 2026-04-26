"""
Real-time object detection + tracking using RF-DETR and ByteTrack.
Works on Mac, Windows, Linux.
Supports built-in webcam, USB camera, or phone camera (IP Webcam / DroidCam).

Controls:
  Q / ESC  — quit
  +        — increase confidence threshold
  -        — decrease confidence threshold
  T        — toggle tracking on/off
  L        — toggle labels on/off
  M        — cycle through model sizes (Nano → Small → Large)

Camera options (edit CAMERA_SOURCE below):
  0                              → built-in / default webcam
  1, 2 ...                       → USB camera or DroidCam virtual webcam
  "http://192.168.x.x:8080/video" → Android IP Webcam app
  "rtsp://..."                   → any RTSP stream
"""

import sys
import cv2
import supervision as sv
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRLarge
try:
    from rfdetr import RFDETRXLarge, RFDETR2XLarge
    XL_AVAILABLE = True
except ImportError:
    XL_AVAILABLE = False
from trackers import ByteTrackTracker
from rfdetr.assets.coco_classes import COCO_CLASSES

# ── Config ────────────────────────────────────────────────────────────────────
# Change CAMERA_SOURCE to your phone IP if using IP Webcam app:
# CAMERA_SOURCE = "http://192.168.1.5:8080/video"
CAMERA_SOURCE = 0          # 0 = default webcam

MODEL_SIZE   = "small"     # "nano" | "small" | "large"
CONFIDENCE   = 0.4
SHOW_LABELS  = True
USE_TRACKER  = True
# ─────────────────────────────────────────────────────────────────────────────

MODEL_MAP = {
    "nano":  RFDETRNano,
    "small": RFDETRSmall,
    "large": RFDETRLarge,
}
if XL_AVAILABLE:
    MODEL_MAP["xl"]  = RFDETRXLarge
    MODEL_MAP["2xl"] = RFDETR2XLarge

MODEL_CYCLE = ["nano", "small", "large"] + (["xl", "2xl"] if XL_AVAILABLE else [])


def load_model(size):
    print(f"\nLoading RF-DETR {size.capitalize()} model...")
    return MODEL_MAP[size]()


def open_camera(source):
    """Open camera from index or URL string."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"\n❌  Could not open camera: {source}")
        if isinstance(source, int):
            print("   → Check webcam is connected and permissions are granted.")
        else:
            print("   → Check phone and PC are on the same WiFi network.")
            print("   → Verify the IP address in the IP Webcam app.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


# ── Load model & camera ───────────────────────────────────────────────────────
current_model_idx = MODEL_CYCLE.index(MODEL_SIZE)
model   = load_model(MODEL_SIZE)
tracker = ByteTrackTracker()
cap     = open_camera(CAMERA_SOURCE)

# Annotators
box_annotator   = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=40)

print(f"✅  Camera opened: {CAMERA_SOURCE}")
print("   Q/ESC=quit  +/-=confidence  T=tracker  L=labels  M=switch model\n")

conf        = CONFIDENCE
show_labels = SHOW_LABELS
use_tracker = USE_TRACKER

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame — check camera connection.")
        break

    # Detect
    detections = model.predict(frame, threshold=conf)

    # Track
    if use_tracker:
        detections = tracker.update(detections)

    # Build labels
    labels = []
    for i, class_id in enumerate(detections.class_id):
        name  = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else "?"
        score = detections.confidence[i] if detections.confidence is not None else 0.0
        tid   = (detections.tracker_id[i]
                 if use_tracker and detections.tracker_id is not None else None)
        labels.append(f"#{tid} {name} {score:.0%}" if tid is not None
                      else f"{name} {score:.0%}")

    # Annotate
    annotated = frame.copy()
    if use_tracker and detections.tracker_id is not None:
        annotated = trace_annotator.annotate(annotated, detections)
    annotated = box_annotator.annotate(annotated, detections)
    if show_labels and labels:
        annotated = label_annotator.annotate(annotated, detections, labels)

    # HUD
    current_size = MODEL_CYCLE[current_model_idx]
    hud = (f"Model: RF-DETR {current_size}  |  "
           f"Objects: {len(detections)}  |  Conf: {conf:.2f}  |  "
           f"Tracker: {'ON' if use_tracker else 'OFF'}  |  "
           f"Labels: {'ON' if show_labels else 'OFF'}")
    cv2.putText(annotated, hud, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2, cv2.LINE_AA)

    cv2.imshow("RF-DETR Real-Time Tracker", annotated)

    # Key handling
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):                    # Q / ESC → quit
        break
    elif key in (ord("+"), ord("=")):
        conf = min(conf + 0.05, 0.95)
        print(f"Confidence: {conf:.2f}")
    elif key == ord("-"):
        conf = max(conf - 0.05, 0.05)
        print(f"Confidence: {conf:.2f}")
    elif key == ord("t"):
        use_tracker = not use_tracker
        tracker = ByteTrackTracker()
        print(f"Tracker: {'ON' if use_tracker else 'OFF'}")
    elif key == ord("l"):
        show_labels = not show_labels
        print(f"Labels: {'ON' if show_labels else 'OFF'}")
    elif key == ord("m"):                        # cycle model size
        current_model_idx = (current_model_idx + 1) % len(MODEL_CYCLE)
        new_size = MODEL_CYCLE[current_model_idx]
        model = load_model(new_size)
        tracker = ByteTrackTracker()
        print(f"Switched to RF-DETR {new_size}")

cap.release()
cv2.destroyAllWindows()
print("Done.")
