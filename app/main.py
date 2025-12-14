import cv2
import numpy as np
import ArducamDepthCamera as ac
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MAX_DISTANCE = 1500
CONFIDENCE_THRESHOLD = 60
YOLO_CONF = 0.5
IP_CAM_URL = "http://10.113.219.223:8080/video"  # Replace with your IP webcam URL
# ----------------------------------------

# Load YOLO
model = YOLO("yolov11n.pt")

# ---------- INIT CAMERAS ----------
rgb_cam = cv2.VideoCapture(IP_CAM_URL)
if not rgb_cam.isOpened():
    print("❌ Cannot open IP webcam")
    exit(1)

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

cv2.namedWindow("RGB + Depth", cv2.WINDOW_AUTOSIZE)

# ---------- MAIN LOOP ----------
while True:
    ret, rgb_frame = rgb_cam.read()
    if not ret:
        continue

    frame = depth_cam.requestFrame(2000)
    if frame is None or not isinstance(frame, ac.DepthData):
        continue

    depth = frame.depth_data.copy()
    conf = frame.confidence_data
    depth[(conf < CONFIDENCE_THRESHOLD) | (depth <= 0) | (depth > MAX_DISTANCE)] = 0

    # YOLO on RGB
    results = model(rgb_frame, conf=YOLO_CONF, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            h, w = depth.shape
            x1 = max(0, min(x1, w-1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h-1))
            y2 = max(0, min(y2, h))

            roi = depth[y1:y2, x1:x2]
            roi = roi[roi > 0]
            if roi.size == 0:
                continue

            distance = int(np.percentile(roi, 30))

            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rgb_frame, f"{label} {distance}mm",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("RGB + Depth", rgb_frame)
    depth_cam.releaseFrame(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------- CLEANUP ----------
rgb_cam.release()
depth_cam.stop()
depth_cam.close()
cv2.destroyAllWindows()
