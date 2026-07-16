from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

from image_processing import preprocess_face
from predictor import predict_emotion
from face_detector import detect_face


app = FastAPI(
    title="Face Emotion Recognition API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "Emotion Recognition API Running 🚀"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # image bytes read
    image_bytes = await file.read()


    # bytes -> numpy array
    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )


    # numpy -> OpenCV image
    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if image is None:
        return {
            "error": "Image read nahi ho rahi"
        }


    # Face detection
    face = detect_face(image)


    if face is None:
        return {
            "error": "Face detect nahi hua"
        }


    # Face preprocessing
    processed_face = preprocess_face(face)


    # CNN prediction
    result = predict_emotion(processed_face)


    return result