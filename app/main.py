import ArducamDepthCamera as ac
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MAX_DISTANCE = 1200              # keep LOW for clarity
CONFIDENCE_THRESHOLD = 80        # very strict
YOLO_CONF = 0.5
ALPHA = 0.8                      # strong temporal smoothing
# ----------------------------------------

# ✅ VALID YOLO MODEL
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

        depth = frame.depth_data.copy()
        conf = frame.confidence_data

        # ---------- HARD FILTERING ----------
        depth[(conf < CONFIDENCE_THRESHOLD) |
              (depth <= 0) |
              (depth > MAX_DISTANCE)] = 0

        # Median filter
        depth = cv2.medianBlur(depth, 7)

        # ---------- TEMPORAL FILTER ----------
        if prev_depth is None:
            prev_depth = depth
        else:
            depth = cv2.addWeighted(depth, ALPHA, prev_depth, 1 - ALPHA, 0)
            prev_depth = depth

        # ---------- VISUALIZATION ----------
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)

        depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)

        # ---------- YOLO (OPTIONAL, DEBUG ONLY) ----------
        results = model(depth_vis, conf=YOLO_CONF, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                roi = depth[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                distance = np.percentile(roi, 25)

                cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(
                    depth_vis,
                    f"{label} {int(distance)}mm",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

        cv2.imshow("preview", depth_vis)

        cam.releaseFrame(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
