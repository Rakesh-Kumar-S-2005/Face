import cv2
import os
from time import sleep
# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Ask user name dynamically
person_name = input("Enter person's name: ").strip()

# Create dataset directory
dataset_path = os.path.join("dataset", person_name)
os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)

img_count = 0
MAX_IMAGES = 100  # change as needed

print("[INFO] Starting face capture... Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        img_count += 1

        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (300, 300))

        img_path = os.path.join(dataset_path, f"{img_count}.jpg")
        cv2.imwrite(img_path, face_img)

        # Draw rectangle & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{person_name} [{img_count}]",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Dataset Capture", frame)
    sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q') or img_count >= MAX_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()

print(f"[INFO] Dataset collected for {person_name}: {img_count} images")
