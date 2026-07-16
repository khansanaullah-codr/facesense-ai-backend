import cv2

from face_detector import detect_face
from image_processing import preprocess_face
from predictor import predict_emotion


image = cv2.imread("smile_real.jpg")


face = detect_face(image)


if face is None:
    print("❌ Face nahi mila")
    exit()


processed = preprocess_face(face)


result = predict_emotion(processed)


print(result)