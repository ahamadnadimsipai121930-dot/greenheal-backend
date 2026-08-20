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

# --- EXACT REMEDIES DATABASE FOR ALL 64 CLASSES ---
REMEDIES_DB = {
    # --- BEAN ---
    "bean_angular_leaf_spot": {
        "treatment": "Apply copper hydroxide or Mancozeb fungicide at early onset.",
        "prevention": "Practice 2-year crop rotation and plant certified disease-free seeds."
    },
    "bean_rust": {
        "treatment": "Spray sulfur, chlorothalonil, or triazole fungicides upon detecting rust spots.",
        "prevention": "Destroy crop debris after harvest and plant rust-resistant bean varieties."
    },
    "healthy_bean": {
        "treatment": "No treatment needed! Your bean crop is healthy.",
        "prevention": "Maintain consistent irrigation and balanced soil nutrients."
    },

    # --- CORN / MAIZE ---
    "corn_cercospora_leaf_spot": {
        "treatment": "Apply foliar fungicides such as azoxystrobin or pyraclostrobin.",
        "prevention": "Rotate crops for at least 1-2 years and practice deep tillage."
    },
    "corn_common_rust": {
        "treatment": "Apply triazole or strobilurin-based fungicides upon early detection.",
        "prevention": "Plant rust-resistant hybrids and avoid late planting."
    },
    "corn_gray_leaf_spot": {
        "treatment": "Spray fungicides like propiconazole or azoxystrobin before lesions reach upper leaves.",
        "prevention": "Practice tillage to bury infected residue and rotate crops."
    },
    "corn_northern_leaf_blight": {
        "treatment": "Apply protective fungicides during the tasseling stage.",
        "prevention": "Utilize resistant crop hybrids and manage crop residue."
    },
    "healthy_corn": {
        "treatment": "No treatment needed! Your corn crop is healthy.",
        "prevention": "Keep fields weed-free and supply balanced nitrogen fertilizer."
    },

    # --- COTTON ---
    "cotton_aphids": {
        "treatment": "Spray neem oil, insecticidal soap, or Acetamiprid on leaf undersides.",
        "prevention": "Avoid excessive nitrogen fertilization and encourage natural predators like ladybugs."
    },
    "cotton_army_worm": {
        "treatment": "Apply bio-pesticides like Bacillus thuringiensis (Bt) or Emamectin benzoate.",
        "prevention": "Set up pheromone traps and clear weeds around field borders."
    },
    "cotton_bacterial_blight": {
        "treatment": "Spray copper oxychloride mixed with Streptocycline.",
        "prevention": "Use certified disease-free seeds and avoid field work when plants are wet."
    },
    "cotton_powdery_mildew": {
        "treatment": "Apply wettable sulfur or Hexaconazole fungicides.",
        "prevention": "Maintain optimal spacing to promote airflow and reduce ambient humidity."
    },
    "cotton_target_spot": {
        "treatment": "Apply fungicides containing azoxystrobin or pyraclostrobin.",
        "prevention": "Manage crop canopy density and rotate crops regularly."
    },
    "healthy_cotton": {
        "treatment": "No treatment needed! Your cotton crop is in good health.",
        "prevention": "Monitor regularly for pests and avoid over-fertilizing with nitrogen."
    },

    # --- CUCUMBER ---
    "diseased_cucumber": {
        "treatment": "Apply broad-spectrum copper-based fungicides or bio-fungicides like Trichoderma.",
        "prevention": "Improve soil drainage, prune old leaves, and avoid overhead watering."
    },
    "healthy_cucumber": {
        "treatment": "No treatment needed! Your cucumber plants look healthy.",
        "prevention": "Water directly at the soil level and provide adequate trellis support."
    },

    # --- GROUNDNUT / PEANUT ---
    "groundnut_early_leaf_spot": {
        "treatment": "Spray Mancozeb or Chlorothalonil early in the season.",
        "prevention": "Rotate groundnuts with non-legume crops to break the disease cycle."
    },
    "groundnut_late_leaf_spot": {
        "treatment": "Apply Carbendazim or Tebuconazole upon lesion appearance.",
        "prevention": "Destroy crop residues after harvest to kill overwintering fungal spores."
    },
    "groundnut_nutrition_deficiency": {
        "treatment": "Apply balanced NPK fertilizers and micronutrients like Gypsum, Zinc, and Iron.",
        "prevention": "Conduct soil testing before planting to address specific mineral gaps."
    },
    "groundnut_rust": {
        "treatment": "Spray Hexaconazole or Mancozeb at the first sign of rust pustules.",
        "prevention": "Plant rust-resistant cultivars and destroy volunteer groundnut plants."
    },
    "healthy_groundnut": {
        "treatment": "No treatment needed! Your groundnut crop is healthy.",
        "prevention": "Maintain adequate soil moisture and apply gypsum during pod formation."
    },

    # --- GUAVA ---
    "guava_anthracnose": {
        "treatment": "Prune dead twigs and spray copper oxychloride or Carbendazim.",
        "prevention": "Ensure proper canopy pruning for sun exposure and avoid fruit physical damage."
    },
    "guava_fruit_fly": {
        "treatment": "Hang methyl eugenol protein bait traps and spray Spinosad or Neem oil.",
        "prevention": "Collect and deeply bury fallen fruits to break the insect life cycle."
    },
    "healthy_guava": {
        "treatment": "No treatment needed! Your guava tree is thriving.",
        "prevention": "Prune annually and maintain balanced drip irrigation."
    },

    # --- LEMON / CITRUS ---
    "lemon_anthracnose": {
        "treatment": "Prune infected twigs and apply broad-spectrum copper fungicides.",
        "prevention": "Ensure good canopy ventilation and avoid leaf wetness during watering."
    },
    "lemon_bacterial_blight": {
        "treatment": "Spray copper hydroxide mixed with streptomycin sulfate.",
        "prevention": "Prune diseased branches during dry weather and sanitize pruning tools."
    },
    "lemon_citrus_canker": {
        "treatment": "Spray copper-based bactericides at bud break and canopy expansion.",
        "prevention": "Control Asian citrus psyllid and leafminers to prevent leaf wounds."
    },
    "lemon_curl_virus": {
        "treatment": "Control vector insect pests (aphids and whiteflies) using Imidacloprid or neem oil.",
        "prevention": "Plant certified virus-free rootstocks and rogue out infected young shoots."
    },
    "lemon_deficiency": {
        "treatment": "Apply micronutrient sprays containing Zinc, Iron, Manganese, and Magnesium.",
        "prevention": "Maintain optimal soil pH (6.0 - 7.0) to ensure maximum nutrient uptake."
    },
    "lemon_dry_leaf": {
        "treatment": "Adjust watering schedule; inspect root system for rot and apply Trichoderma.",
        "prevention": "Ensure soil drains well and protect trees from hot, dry wind conditions."
    },
    "lemon_sooty_mould": {
        "treatment": "Wash leaves with insecticidal soap; treat honeydew-secreting pests like scale insects.",
        "prevention": "Control sap-sucking insects (aphids, scales, whiteflies) that produce honeydew."
    },
    "lemon_spider_mites": {
        "treatment": "Apply miticides, horticultural oils, or neem oil spray.",
        "prevention": "Maintain adequate ambient humidity around the canopy and reduce dust."
    },
    "healthy_lemon": {
        "treatment": "No treatment needed! Your lemon tree is healthy.",
        "prevention": "Ensure full sunlight, well-draining soil, and seasonal organic fertilizer."
    },

    # --- POTATO ---
    "potato_early_blight": {
        "treatment": "Apply Mancozeb, Chlorothalonil, or copper fungicides at early onset.",
        "prevention": "Rotate crops with non-solanaceous plants every 2-3 years."
    },
    "potato_late_blight": {
        "treatment": "Apply systemic fungicides like Metalaxyl or Dimethomorph immediately.",
        "prevention": "Use certified seed tubers and avoid overhead sprinkler watering."
    },
    "healthy_potato": {
        "treatment": "No treatment needed! Your potato crop is in great condition.",
        "prevention": "Maintain proper soil hilling and avoid waterlogging in the root zone."
    },

    # --- PUMPKIN ---
    "pumpkin_bacterial_leaf_spot": {
        "treatment": "Spray copper-based bactericides early in the morning.",
        "prevention": "Avoid working in fields when vines are wet and practice crop rotation."
    },
    "pumpkin_downy_mildew": {
        "treatment": "Apply fungicides containing Mancozeb, Fosetyl-Al, or copper compounds.",
        "prevention": "Plant in full sun and space vines adequately to improve air flow."
    },
    "pumpkin_mosaic_disease": {
        "treatment": "Remove infected vines and control aphid vectors using neem oil.",
        "prevention": "Control weeds around the field border that serve as virus reservoirs."
    },
    "pumpkin_powdery_mildew": {
        "treatment": "Spray sulfur, potassium bicarbonate, or Hexaconazole fungicides.",
        "prevention": "Plant resistant pumpkin cultivars and remove field weeds."
    },
    "healthy_pumpkin": {
        "treatment": "No treatment needed! Your pumpkin vines are healthy.",
        "prevention": "Mulch around vines and use drip irrigation to keep foliage dry."
    },

    # --- RICE ---
    "diseased_rice": {
        "treatment": "Apply targeted fungicides like Tricyclazole (for blast) or Propiconazole.",
        "prevention": "Avoid excessive nitrogen fertilization and maintain optimal field water depth."
    },
    "healthy_rice": {
        "treatment": "No treatment needed! Your rice field is in great condition.",
        "prevention": "Maintain proper water management and balanced nutrient application."
    },

    # --- SUGARCANE ---
    "sugarcane_bacterial_blight": {
        "treatment": "Spray copper oxychloride and rogue out severely affected plant clumps.",
        "prevention": "Plant disease-free setts and ensure effective field drainage."
    },
    "sugarcane_mosaic": {
        "treatment": "Rogue out infected plants and control aphid vectors using insecticides.",
        "prevention": "Use mosaic-resistant sugarcane varieties and clean seed setts."
    },
    "sugarcane_red_rot": {
        "treatment": "Rogue out infected stalks and treat seed setts with Carbendazim before planting.",
        "prevention": "Maintain good field drainage, avoid waterlogging, and rotate crops."
    },
    "sugarcane_rust": {
        "treatment": "Spray Mancozeb or Propiconazole upon detecting rust pustules.",
        "prevention": "Grow rust-resistant sugarcane cultivars."
    },
    "sugarcane_yellow_leaf_disease": {
        "treatment": "Control aphid vectors using systemic insecticides. Remove affected clumps.",
        "prevention": "Plant tissue-cultured virus-free seed setts."
    },
    "healthy_sugarcane": {
        "treatment": "No treatment needed! Your sugarcane crop is thriving.",
        "prevention": "Keep fields weed-free and maintain adequate irrigation during tillering."
    },

    # --- WHEAT ---
    "wheat_aphid": {
        "treatment": "Spray Dimethoate or Imidacloprid if aphid population exceeds threshold.",
        "prevention": "Encourage natural predators like ladybird beetles and lacewings."
    },
    "wheat_black_rust": {
        "treatment": "Apply triazole fungicides like Tebuconazole or Propiconazole promptly.",
        "prevention": "Destroy alternate host plants (barberry) near fields."
    },
    "wheat_blast": {
        "treatment": "Apply Tebuconazole + Trifloxystrobin or Mancozeb sprays.",
        "prevention": "Use certified disease-free seeds and avoid excessive late-season nitrogen."
    },
    "wheat_brown_rust": {
        "treatment": "Apply Propiconazole or Tebuconazole fungicides.",
        "prevention": "Plant rust-resistant wheat varieties."
    },
    "wheat_common_root_rot": {
        "treatment": "Treat seeds with fungicides like Carboxin or Thiram before sowing.",
        "prevention": "Rotate with non-cereal crops and avoid soil compaction."
    },
    "wheat_fusarium_head_blight": {
        "treatment": "Spray Tebuconazole or Metconazole during early flowering stage.",
        "prevention": "Avoid planting wheat directly after corn in crop rotation."
    },
    "wheat_leaf_blight": {
        "treatment": "Apply Propiconazole or Mancozeb fungicides early in the infection.",
        "prevention": "Ensure balanced fertilization and practice proper field sanitation."
    },
    "wheat_mildew": {
        "treatment": "Apply wettable sulfur or Triadimefon fungicides.",
        "prevention": "Avoid high seed rates to improve air circulation in the canopy."
    },
    "wheat_mite": {
        "treatment": "Spray wettable sulfur or miticides like Propargite.",
        "prevention": "Destroy volunteer wheat plants before the new sowing season."
    },
    "wheat_septoria": {
        "treatment": "Apply strobilurin or triazole fungicides like Azoxystrobin.",
        "prevention": "Practice stubble management and multi-year crop rotation."
    },
    "wheat_smut": {
        "treatment": "Treat seeds with systemic fungicides like Carboxin or Tebuconazole.",
        "prevention": "Use certified disease-free seed stock."
    },
    "wheat_stem_fly": {
        "treatment": "Apply Chlorpyrifos or Thiamethoxam during early vegetative stage.",
        "prevention": "Adjust sowing dates to avoid peak stem fly activity."
    },
    "wheat_tan_spot": {
        "treatment": "Apply triazole-based fungicides if lesions appear on flag leaves.",
        "prevention": "Incorporate or destroy crop residues after harvest."
    },
    "wheat_yellow_rust": {
        "treatment": "Apply Propiconazole or Tebuconazole at first sign of yellow stripes.",
        "prevention": "Plant resistant wheat cultivars."
    },
    "healthy_wheat": {
        "treatment": "No treatment needed! Your wheat field is in great health.",
        "prevention": "Ensure proper irrigation scheduling during critical growth stages."
    }
}

def get_treatment_details(disease_label: str):
    # Normalize input string to lower-case key format
    key = disease_label.strip().lower()
    
    # Direct exact lookup
    if key in REMEDIES_DB:
        return REMEDIES_DB[key]
    
    # Partial fallback matching
    for db_key in REMEDIES_DB.keys():
        if db_key in key or key in db_key:
            return REMEDIES_DB[db_key]

    # Healthy fallback
    if "healthy" in key:
        return {
            "treatment": "No treatment needed! The plant is healthy.",
            "prevention": "Maintain standard soil health, proper sunlight, and routine watering."
        }
    
    # Generic fallback
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
    
    # Fetch treatments
    remedy_info = get_treatment_details(predicted_class)
    clean_disease_name = predicted_class.replace('_', ' ').title()

    # Translate response if non-English
    if lang != "en":
        try:
            translator = GoogleTranslator(source='en', target=lang)
            clean_disease_name = translator.translate(clean_disease_name)
            remedy_info["treatment"] = translator.translate(remedy_info["treatment"])
            remedy_info["prevention"] = translator.translate(remedy_info["prevention"])
        except Exception:
            pass

    return {
        "prediction": clean_disease_name,
        "confidence_score": f"{confidence:.2f}%",
        "treatment": remedy_info["treatment"],
        "prevention": remedy_info["prevention"]
    }
