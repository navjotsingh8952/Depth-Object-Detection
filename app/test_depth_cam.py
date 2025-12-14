import ArducamDepthCamera as ac
import cv2
import numpy as np

MAX_DISTANCE = 1500  # mm

def main():
    print("=== Arducam Depth Camera Test ===")
    print("SDK version:", ac.__version__)

    cam = ac.ArducamCamera()

    # IMPORTANT: USB connection
    ret = cam.open(ac.Connection.USB, 0)
    print("Open return:", ret)
    if ret != 0:
        print("❌ Failed to open camera")
        return

    ret = cam.start(ac.FrameType.DEPTH)
    print("Start return:", ret)
    if ret != 0:
        print("❌ Failed to start depth stream")
        cam.close()
        return

    cam.setControl(ac.Control.RANGE, MAX_DISTANCE)
    depth_range = cam.getControl(ac.Control.RANGE)
    print("Depth range:", depth_range)

    cv2.namedWindow("Depth Preview", cv2.WINDOW_AUTOSIZE)

    while True:
        frame = cam.requestFrame(2000)
        if frame is None:
            print("No frame")
            continue

        if not isinstance(frame, ac.DepthData):
            cam.releaseFrame(frame)
            continue

        depth = frame.depth_data.copy()
        conf = frame.confidence_data

        # Remove invalid depth
        depth[(depth <= 0) | (depth > MAX_DISTANCE)] = 0

        # Normalize to 8-bit (SAFE)
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)

        # Apply colormap
        depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)

        # Confidence mask
        depth_vis[conf < 30] = (0, 0, 0)

        cv2.imshow("Depth Preview", depth_vis)
        cam.releaseFrame(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cam.close()
    cv2.destroyAllWindows()
    print("=== Test Finished ===")


if __name__ == "__main__":
    main()
