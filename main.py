import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from deep_translator import GoogleTranslator

app = FastAPI(title="GreenHeal AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "greenheal_mobilenetv2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

REMEDIES_DB = {
    "wheat_brown_rust": {
        "treatment": "Apply triazole-based fungicides upon early detection.",
        "prevention": "Plant rust-resistant seed varieties and rotate crops."
    }
}

def get_treatment_details(disease_label: str):
    key = disease_label.lower().replace(" ", "_")
    if key in REMEDIES_DB:
        return REMEDIES_DB[key]
    readable_name = disease_label.replace("_", " ").title()
    return {
        "treatment": f"Isolate infected plants and apply recommended treatments for {readable_name}.",
        "prevention": "Ensure adequate spacing for airflow and avoid overhead watering."
    }

@app.get("/")
def health_check():
    return {"status": "online"}

@app.post("/predict")
async def predict_leaf(lang: str = Form("en"), file: UploadFile = File(...)):
    # Process Image
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(tf.keras.utils.img_to_array(img), axis=0))
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]) * 100)
    
    # Get standard English remedies
    remedy_info = get_treatment_details(predicted_class)
    clean_disease_name = predicted_class.replace('_', ' ').title()

    # Translate if language is not English
    if lang != "en":
        try:
            translator = GoogleTranslator(source='en', target=lang)
            clean_disease_name = translator.translate(clean_disease_name)
            remedy_info["treatment"] = translator.translate(remedy_info["treatment"])
            remedy_info["prevention"] = translator.translate(remedy_info["prevention"])
        except Exception:
            pass # Fallback to English if translation fails

    return {
        "prediction": clean_disease_name,
        "confidence_score": f"{confidence:.2f}%",
        "treatment": remedy_info["treatment"],
        "prevention": remedy_info["prevention"]
    }