import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from deep_translator import GoogleTranslator

app = FastAPI(title="GreenHeal AI Backend")

# Allow Chrome and mobile browsers to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "greenheal_mobilenetv2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# --- COMPREHENSIVE DISEASE DATABASE ---
REMEDIES_DB = {
    # Wheat & Cotton
    "wheat_brown_rust": {
        "treatment": "Apply triazole-based fungicides (like Tebuconazole) upon early detection. Remove heavily infected leaves.",
        "prevention": "Plant rust-resistant seed varieties, ensure proper crop rotation, and avoid excessive nitrogen fertilizer."
    },
    "cotton_aphids": {
        "treatment": "1. Spray neem oil or insecticidal soap on the underside of the leaves. 2. Release natural predators like Ladybugs. 3. For severe infestations, use Acetamiprid sprays.",
        "prevention": "Avoid over-fertilizing with nitrogen, which attracts aphids. Regularly prune the lower canopy to improve airflow."
    },
    
    # Potato
    "potato_early_blight": {
        "treatment": "Apply fungicides containing chlorothalonil or copper-based sprays early in the morning.",
        "prevention": "Practice a 2-3 year crop rotation. Ensure tubers are fully mature before harvesting."
    },
    "potato_late_blight": {
        "treatment": "Urgently apply systemic fungicides like metalaxyl. Destroy severely infected plants immediately.",
        "prevention": "Plant certified disease-free seed potatoes. Destroy volunteer potatoes and cull piles."
    },
    
    # Tomato
    "tomato_early_blight": {
        "treatment": "Prune infected lower leaves. Apply copper fungicide or chlorothalonil every 7-10 days.",
        "prevention": "Use mulch to prevent soil spores from splashing onto leaves. Water at the base, not overhead."
    },
    "tomato_late_blight": {
        "treatment": "Apply systemic fungicides immediately. Remove and burn heavily infected plants to stop the spread.",
        "prevention": "Ensure wide spacing for ventilation. Avoid overhead watering and keep foliage dry."
    },
    "tomato_leaf_mold": {
        "treatment": "Improve airflow immediately. Apply fungicides like chlorothalonil if the infection is severe.",
        "prevention": "Maintain greenhouse humidity below 85%. Stake and prune plants to improve air circulation."
    },
    
    # Corn
    "corn_common_rust": {
        "treatment": "Apply foliar fungicides early in the season if pustules are observed on multiple leaves.",
        "prevention": "Plant resistant hybrids. Plant early to avoid peak rust conditions later in the season."
    },
    "corn_northern_leaf_blight": {
        "treatment": "Use protective fungicides during the tasseling stage if lesions are actively spreading.",
        "prevention": "Till crop residue deeply into the soil after harvest to destroy overwintering fungi."
    },
    
    # Apple
    "apple_scab": {
        "treatment": "Spray Captan or sulfur-based fungicides during the pink bud stage.",
        "prevention": "Rake and destroy fallen leaves in autumn where the fungus overwinters. Prune trees to open the canopy."
    },
    "apple_black_rot": {
        "treatment": "Prune out dead or diseased branches. Apply Captan fungicides starting at petal fall.",
        "prevention": "Remove mummified fruit and dead wood from the orchard. Ensure good drainage."
    },
    
    # Healthy (No disease)
    "healthy": {
        "treatment": "No treatment needed! Your plant is growing beautifully.",
        "prevention": "Continue maintaining good soil health, proper watering, and regular monitoring."
    }
}

def get_treatment_details(disease_label: str):
    # Convert prediction to database key format (e.g., "Cotton Aphids" -> "cotton_aphids")
    key = disease_label.lower().replace(" ", "_")
    
    if key in REMEDIES_DB:
        return REMEDIES_DB[key]
    
    # Fallback if the disease is not in our database yet
    readable_name = disease_label.replace("_", " ").title()
    return {
        "treatment": f"Consult a local agricultural expert for specific treatments targeting {readable_name}.",
        "prevention": "Ensure adequate spacing for airflow, avoid overhead watering, and maintain clean soil."
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
