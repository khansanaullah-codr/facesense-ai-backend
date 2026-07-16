import numpy as np

from app.model_loader import model
from app.emotion_labels import EMOTIONS


def predict_emotion(face):

    prediction = model.predict(face, verbose=0)

    index = np.argmax(prediction)

    confidence = float(prediction[0][index])

    return {
        "emotion": EMOTIONS[index],
        "confidence": round(confidence * 100, 2)
    }