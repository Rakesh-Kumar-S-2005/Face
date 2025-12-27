import cv2
import os
import numpy as np
import pickle


dataset_path = "dataset"

faces = []
labels = []
label_map = {}  # name → number
current_label = 0

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces.append(img)
        labels.append(current_label)

    current_label += 1

faces = np.array(faces)
labels = np.array(labels)

print("[INFO] Dataset loaded")
print("Total faces:", len(faces))
print("Labels:", label_map)

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

recognizer.train(faces, labels)

recognizer.save("face_model.yml")

print("[INFO] Training completed & model saved")

with open("labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("[INFO] Label map saved")
