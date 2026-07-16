from tensorflow.keras.models import load_model
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "emotion_model.keras")

model = load_model(MODEL_PATH)

print("✅ Emotion Model Loaded Successfully")