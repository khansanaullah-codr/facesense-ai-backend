import cv2
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# app se ek level upar backend, fir haarcascade folder
cascade_path = os.path.join(
    os.path.dirname(BASE_DIR),
    "haarcascade",
    "haarcascade_frontalface_default.xml"
)


face_cascade = cv2.CascadeClassifier(cascade_path)


if face_cascade.empty():
    print("❌ Haar Cascade load nahi hui:", cascade_path)
else:
    print("✅ Haar Cascade loaded")


def detect_face(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    face = image[y:y+h, x:x+w]

    return face