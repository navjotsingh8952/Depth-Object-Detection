import face_recognition
import pickle
import cv2
import time

TOLERANCE = 0.6

# Load trained encodings
with open("trained_faces.pkl", "rb") as f:
    data = pickle.load(f)

cap = cv2.VideoCapture(0)

total = 0
correct = 0
last_name = None
last_time = 0
COOLDOWN = 1.5  # seconds per face count

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(
            data["encodings"], face_encoding
        )

        best_match_index = distances.argmin()
        name = "Unknown"

        if distances[best_match_index] < TOLERANCE:
            name = data["names"][best_match_index]

        # ----- Accuracy counting (avoid frame spam) -----
        now = time.time()
        if name != last_name or now - last_time > COOLDOWN:
            total += 1
            if name != "Unknown":
                # speak
                ...
            last_name = name
            last_time = now

        # ----- Draw -----
        label = f"{name}"
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Webcam Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()