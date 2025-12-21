import time

import cv2

from face_recognition_module import detect_face
from object_detection_module import detect_object
# from touch import is_touched
from tts import speak
from ultrasonic import Ultrasonic

# ---------- SENSORS ----------
# ultra_left = Ultrasonic(trig=23, echo=24)
# ultra_right = Ultrasonic(trig=27, echo=22)
cap = cv2.VideoCapture(0)

last_mode = None
old_name = None
left = right=0
try:
    while True:
        mode = "FACE"
        # mode = "FACE" if is_touched() else "OBJECT"

        if mode != last_mode:
            speak(f"{mode} mode activated")
            print(f"{mode} mode activated")
            last_mode = mode

        # left = ultra_left.distance_cm()
        # right = ultra_right.distance_cm()
        ret, frame = cap.read()
        if not ret:
            continue
        speak(f"Distance Left: {left}, Right: {right}")
        print(f"Distance Left: {left}, Right: {right}")
        # ---------- OBJECT MODE ----------
        if mode == "OBJECT":
            frame, objects = detect_object(frame)
            if objects:
                speak(f"Detected: {','.join(objects)}")
                print(f"Detected: {','.join(objects)}")

            cv2.imshow("Webcam Object Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # ---------- FACE MODE ----------
        else:
            frame, name = detect_face(frame)
            if name != "Unknown" and name != old_name:
                speak(f"{name} detected")
                print(f"{name} detected")

                old_name = name

            cv2.imshow("Webcam Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        time.sleep(0.3)

except KeyboardInterrupt:
    print("Stopping...")

finally:
    cap.release()
    cv2.destroyAllWindows()
