import cv2
import numpy as np
import ArducamDepthCamera as ac
from ultralytics import YOLO

# ================= CONFIG =================
MAX_DISTANCE = 1500        # mm (keep low for clean depth)
CONFIDENCE_THRESHOLD = 60  # depth confidence
YOLO_CONF = 0.5
# ==========================================

# Load YOLO model (lightweight)
model = YOLO("yolov11n.pt")

# ---------- INIT RGB CAMERA ----------
rgb_cam = cv2.VideoCapture(0)  # USB webcam or Pi camera
rgb_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
rgb_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not rgb_cam.isOpened():
    print("❌ RGB camera not found")
    exit(1)

# ---------- INIT DEPTH CAMERA ----------
depth_cam = ac.ArducamCamera()

if depth_cam.open(ac.Connection.CSI, 0) != 0:
    print("❌ Failed to open depth camera")
    exit(1)

if depth_cam.start(ac.FrameType.DEPTH) != 0:
    print("❌ Failed to start depth camera")
    depth_cam.close()
    exit(1)

depth_cam.setControl(ac.Control.RANGE, MAX_DISTANCE)

print("✅ Cameras initialized")
print("SDK version:", ac.__version__)

cv2.namedWindow("RGB + Depth", cv2.WINDOW_AUTOSIZE)


def clean_depth(depth, conf):
    """Remove noise from depth"""
    depth[(conf < CONFIDENCE_THRESHOLD) |
          (depth <= 0) |
          (depth > MAX_DISTANCE)] = 0
    return depth


# ================= MAIN LOOP =================
while True:
    # ---- RGB FRAME ----
    ret, rgb_frame = rgb_cam.read()
    if not ret:
        print("⚠ RGB frame missing")
        continue

    # ---- DEPTH FRAME ----
    depth_frame = depth_cam.requestFrame(2000)
    if depth_frame is None or not isinstance(depth_frame, ac.DepthData):
        continue

    depth = depth_frame.depth_data.copy()
    conf = depth_frame.confidence_data

    depth = clean_depth(depth, conf)

    # ---- YOLO DETECTION (ON RGB ONLY) ----
    results = model(rgb_frame, conf=YOLO_CONF, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Clamp box inside depth frame
            h, w = depth.shape
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            roi = depth[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Stable distance (ignore zeros)
            roi = roi[roi > 0]
            if roi.size == 0:
                continue

            distance = int(np.percentile(roi, 30))

            # ---- DRAW ----
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                rgb_frame,
                f"{label} {distance}mm",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("RGB + Depth", rgb_frame)
    depth_cam.releaseFrame(depth_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================= CLEANUP =================
rgb_cam.release()
depth_cam.stop()
depth_cam.close()
cv2.destroyAllWindows()
