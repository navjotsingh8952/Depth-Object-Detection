import face_recognition
import pickle
import os

dataset_dir = "dataset"
known_encodings = []
known_names = []

for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)

        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            print(f"No face found in {img_path}")
            continue

        known_encodings.append(encodings[0])
        known_names.append(person)

# Save to disk
data = {"encodings": known_encodings, "names": known_names}
with open("trained_faces.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ Encodings saved!")
