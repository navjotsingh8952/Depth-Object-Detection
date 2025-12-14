import ArducamDepthCamera as ac
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MAX_DISTANCE = 1500  # VERY IMPORTANT
CONFIDENCE_THRESHOLD = 60  # stricter confidence
YOLO_CONF = 0.5
ALPHA = 0.7  # temporal smoothing
# ----------------------------------------

# ✅ Correct YOLO model
model = YOLO("yolov11n.pt")

prev_depth = None


def main():
    global prev_depth

    print("Arducam Depth + Object Detection Demo")
    print("SDK version:", ac.__version__)

    cam = ac.ArducamCamera()

    if cam.open(ac.Connection.CSI, 0) != 0:
        print("Failed to open camera")
        return

    if cam.start(ac.FrameType.DEPTH) != 0:
        print("Failed to start camera")
        cam.close()
        return

    cam.setControl(ac.Control.RANGE, MAX_DISTANCE)
    depth_range = cam.getControl(ac.Control.RANGE)

    cv2.namedWindow("preview", cv2.WINDOW_AUTOSIZE)

    while True:
        frame = cam.requestFrame(2000)

        if frame is None or not isinstance(frame, ac.DepthData):
            continue

        depth_buf = frame.depth_data.copy()
        confidence_buf = frame.confidence_data

        # -------- DEPTH CLEANING --------
        depth_buf[(depth_buf <= 0) | (depth_buf > MAX_DISTANCE)] = 0
        depth_buf = cv2.medianBlur(depth_buf, 5)

        # -------- TEMPORAL FILTER (CRITICAL) --------
        if prev_depth is None:
            prev_depth = depth_buf
        else:
            depth_buf = cv2.addWeighted(
                depth_buf, ALPHA, prev_depth, 1 - ALPHA, 0
            )
            prev_depth = depth_buf

        # -------- VISUALIZATION IMAGE --------
        depth_vis = (depth_buf * (255.0 / depth_range)).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

        # -------- CONFIDENCE MASK + MORPHOLOGY --------
        mask = (confidence_buf >= CONFIDENCE_THRESHOLD).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        depth_vis[mask == 0] = (0, 0, 0)

        # -------- YOLO DETECTION (NO BLUR HERE) --------
        results = model(depth_vis, conf=YOLO_CONF, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(depth_buf.shape[1], x2)
                y2 = min(depth_buf.shape[0], y2)

                roi = depth_buf[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # ✅ Stable distance
                distance = np.percentile(roi, 30)

                cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(
                    depth_vis,
                    f"{label} {int(distance)}mm",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

        # -------- DISPLAY (LIGHT BLUR OK) --------
        display = cv2.GaussianBlur(depth_vis, (3, 3), 0)
        cv2.imshow("preview", display)

        cam.releaseFrame(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
