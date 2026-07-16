import cv2
import numpy as np


def preprocess_face(face):

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (48, 48))

    normalized = resized.astype("float32") / 255.0

    normalized = np.expand_dims(normalized, axis=-1)

    normalized = np.expand_dims(normalized, axis=0)

    return normalized