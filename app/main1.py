import cv2
import numpy as np
import ArducamDepthCamera as ac

# ---------------- CONFIG ----------------
IP_CAM_URL = "0"
MAX_DISTANCE = 4000
CONFIDENCE_VALUE = 30
SSD_CONF = 0.45

# Paths
COCO_NAMES = "./res/coco.names"
CONFIG_PATH = "./res/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
WEIGHTS_PATH = "./res/frozen_inference_graph.pb"
# ----------------------------------------

# Load class names
with open(COCO_NAMES, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# ---------- MobileNet SSD ----------
net = cv2.dnn_DetectionModel(WEIGHTS_PATH, CONFIG_PATH)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# ---------- IP Webcam ----------
cap = cv2.VideoCapture(IP_CAM_URL)

# ---------- Depth Camera ----------
depth_cam = ac.ArducamCamera()

if depth_cam.open(ac.Connection.CSI, 0) != 0:
    print("❌ Failed to open depth camera")
    exit()

if depth_cam.start(ac.FrameType.DEPTH) != 0:
    print("❌ Failed to start depth camera")
    depth_cam.close()
    exit()

depth_cam.setControl(ac.Control.RANGE, MAX_DISTANCE)
depth_range = depth_cam.getControl(ac.Control.RANGE)

print("✅ Cameras initialized")

# ---------- MAIN LOOP ----------
while True:
    ret, rgb = cap.read()
    if not ret:
        continue

    frame = depth_cam.requestFrame(2000)
    if frame is None or not isinstance(frame, ac.DepthData):
        continue

    depth = frame.depth_data.copy()
    confidence = frame.confidence_data

    # Clean depth
    depth[(confidence < CONFIDENCE_VALUE) | (depth <= 0) | (depth > MAX_DISTANCE)] = 0

    # Resize depth to match RGB
    depth_resized = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))

    # ---------- Object Detection ----------
    classIds, confs, boxes = net.detect(rgb, confThreshold=SSD_CONF)

    if len(classIds) != 0:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), boxes):

            x, y, w, h = box
            cx = x + w // 2
            cy = y + h // 2

            # Safe depth lookup
            if cy >= depth_resized.shape[0] or cx >= depth_resized.shape[1]:
                continue

            distance = depth_resized[cy, cx]
            label = classNames[classId - 1]

            cv2.rectangle(rgb, box, (0, 255, 0), 2)
            cv2.circle(rgb, (cx, cy), 4, (0, 0, 255), -1)

            cv2.putText(
                rgb,
                f"{label} {int(distance)}mm",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    cv2.imshow("IP Cam + Depth Object Detection", rgb)

    depth_cam.releaseFrame(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------- CLEANUP ----------
cap.release()
depth_cam.stop()
depth_cam.close()
cv2.destroyAllWindows()
