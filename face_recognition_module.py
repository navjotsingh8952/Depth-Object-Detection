import pickle

import cv2
import face_recognition

with open("trained_faces.pkl", "rb") as f:
    data = pickle.load(f)
TOLERANCE = 0.6
COOLDOWN = 1.5  # seconds per face count
last_name = None
last_time = 0


def detect_face(frame):
    rgb = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    if not face_encodings:
        return None

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(
            data["encodings"], face_encoding
        )

        best_match_index = distances.argmin()
        name = "Unknown"

        if distances[best_match_index] < TOLERANCE:
            name = data["names"][best_match_index]

        # ----- Draw -----
        label = f"{name}"
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame, name
    return frame, "Unknown"
