import face_recognition
import pickle
import cv2

# Load encodings
with open("trained_faces.pkl", "rb") as f:
    data = pickle.load(f)

# Load new image
image = face_recognition.load_image_file("unknown.jpg")
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(data["encodings"], face_encoding)
    name = "Unknown"

    if True in matches:
        matched_idxs = [i for i, m in enumerate(matches) if m]
        name = data["names"][matched_idxs[0]]

    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image_bgr, name, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("Result", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
