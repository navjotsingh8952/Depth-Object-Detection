import face_recognition
import pickle
import os

DATASET_DIR = "dataset"
TOLERANCE = 0.6   # lower = stricter

# Load trained data
with open("trained_faces.pkl", "rb") as f:
    data = pickle.load(f)

total = 0
correct = 0

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)

        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            print(f"❌ No face found: {img_path}")
            continue

        face_encoding = encodings[0]

        matches = face_recognition.compare_faces(
            data["encodings"], face_encoding, tolerance=TOLERANCE
        )

        name = "Unknown"
        if True in matches:
            name = data["names"][matches.index(True)]

        total += 1
        if name == person_name:
            correct += 1
            result = "✅"
        else:
            result = "❌"

        print(f"{result} Image: {img_name} | Actual: {person_name} | Predicted: {name}")

# Accuracy
accuracy = (correct / total) * 100 if total > 0 else 0
print("\n==============================")
print(f"Total Images   : {total}")
print(f"Correct        : {correct}")
print(f"Accuracy       : {accuracy:.2f}%")
print("==============================")
