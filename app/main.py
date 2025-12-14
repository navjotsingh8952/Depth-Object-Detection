import cv2
import numpy as np
import ArducamDepthCamera as ac
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MAX_DISTANCE = 4000
CONFIDENCE_THRESHOLD = 30
YOLO_CONF = 0.5
# ----------------------------------------

# Load YOLO model
model = YOLO("yolov11n.pt")  # lightweight, good for Raspberry Pi


def getPreviewRGB(preview: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    preview = np.nan_to_num(preview)
    preview[confidence < CONFIDENCE_THRESHOLD] = (0, 0, 0)
    return preview


def main():
    print("Arducam Depth + Object Detection Demo")
    print("SDK version:", ac.__version__)

    cam = ac.ArducamCamera()

    ret = cam.open(ac.Connection.CSI, 0)
    if ret != 0:
        print("Failed to open camera. Error code:", ret)
        return

    ret = cam.start(ac.FrameType.DEPTH)
    if ret != 0:
        print("Failed to start camera. Error code:", ret)
        cam.close()
        return

    cam.setControl(ac.Control.RANGE, MAX_DISTANCE)
    depth_range = cam.getControl(ac.Control.RANGE)

    info = cam.getCameraInfo()
    print(f"Resolution: {info.width}x{info.height}")

    cv2.namedWindow("preview", cv2.WINDOW_AUTOSIZE)

    while True:
        frame = cam.requestFrame(2000)

        if frame is not None and isinstance(frame, ac.DepthData):
            depth_buf = frame.depth_data.copy()
            confidence_buf = frame.confidence_data

            # ---- CLEAN DEPTH ----
            depth_buf[(depth_buf <= 0) | (depth_buf > MAX_DISTANCE)] = 0
            depth_buf = cv2.medianBlur(depth_buf, 5)

            # ---- DEPTH VIS ----
            depth_vis = (depth_buf * (255.0 / depth_range)).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

            mask = confidence_buf >= CONFIDENCE_THRESHOLD
            depth_vis[~mask] = (0, 0, 0)

            depth_vis = cv2.GaussianBlur(depth_vis, (5, 5), 0)

            # ---------------- YOLO DETECTION ----------------
            results = model(depth_vis, conf=YOLO_CONF, verbose=False)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]

                    # Clamp box
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(depth_buf.shape[1], x2)
                    y2 = min(depth_buf.shape[0], y2)

                    roi = depth_buf[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    distance = np.mean(roi)

                    # Draw bounding box
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
            # ------------------------------------------------

            cv2.imshow("preview", depth_vis)
            cam.releaseFrame(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
