
from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

DISEASE_MODEL_PATH = (
    MODEL_DIR / "mobilenetv3_full39_best.pth"
)

LEAF_VALIDATOR_PATH = (
    MODEL_DIR / "leaf_validator_v3_best.pth"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"ML Device: {device}")


# ============================================================
# THRESHOLDS
# ============================================================

LEAF_THRESHOLD = 0.50
DISEASE_THRESHOLD = 0.75


# ============================================================
# 39 CLASS NAMES
# ============================================================

DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Background_without_leaves",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD LEAF VALIDATOR
# ============================================================

print("Loading leaf validator...")

leaf_validator = models.mobilenet_v3_small(weights=None)

leaf_validator.classifier[3] = torch.nn.Linear(
    leaf_validator.classifier[3].in_features,
    2
)

leaf_checkpoint = torch.load(
    LEAF_VALIDATOR_PATH,
    map_location=device,
    weights_only=True
)

leaf_validator.load_state_dict(leaf_checkpoint)
leaf_validator.to(device)
leaf_validator.eval()

print("Leaf validator loaded.")


# ============================================================
# LOAD FINAL 39-CLASS DISEASE MODEL
# ============================================================

print("Loading final 39-class disease model...")

disease_model = models.mobilenet_v3_large(weights=None)

disease_model.classifier[3] = torch.nn.Linear(
    disease_model.classifier[3].in_features,
    len(DISEASE_CLASSES)
)

disease_checkpoint = torch.load(
    DISEASE_MODEL_PATH,
    map_location=device,
    weights_only=False
)

disease_model.load_state_dict(
    disease_checkpoint["model_state_dict"]
)

disease_model.to(device)
disease_model.eval()

print("Final 39-class disease model loaded.")


# ============================================================
# LEAF VALIDATION
# ============================================================

def validate_leaf(image: Image.Image):

    image_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        outputs = leaf_validator(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        # Class 0 = leaf
        leaf_probability = probabilities[0][0].item()

    if leaf_probability < LEAF_THRESHOLD:

        return {
            "is_leaf": False,
            "leaf_confidence": leaf_probability
        }

    return {
        "is_leaf": True,
        "leaf_confidence": leaf_probability
    }


# ============================================================
# 39-CLASS DISEASE PREDICTION
# ============================================================

def predict_disease(image: Image.Image):

    image_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        outputs = disease_model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    confidence = confidence.item()
    predicted_index = predicted_index.item()

    predicted_class = DISEASE_CLASSES[predicted_index]

    # LOW CONFIDENCE
    if confidence < DISEASE_THRESHOLD:

        return {
            "status": "uncertain",
            "disease": None,
            "confidence": confidence,
            "predicted_class": predicted_class,
            "message": (
                "The model confidence is below the required "
                "threshold. Please upload a clearer image "
                "or consult an agricultural expert."
            )
        }

    # BACKGROUND / NO LEAF
    if predicted_class == "Background_without_leaves":

        return {
            "status": "invalid_input",
            "disease": None,
            "confidence": confidence,
            "predicted_class": predicted_class,
            "message": (
                "No clear plant leaf was detected. "
                "Please upload a clear image of a plant leaf."
            )
        }

    # HEALTHY PLANT
    if predicted_class.endswith("___healthy"):

        crop_name = predicted_class.split("___")[0]

        return {
            "status": "healthy",
            "disease": None,
            "confidence": confidence,
            "predicted_class": predicted_class,
            "crop": crop_name,
            "message": (
                f"No disease was detected in the uploaded "
                f"{crop_name} leaf image."
            )
        }

    # DISEASE DETECTED
    return {
        "status": "prediction",
        "disease": predicted_class,
        "confidence": confidence
    }


# ============================================================
# COMPLETE PREDICTION PIPELINE
# ============================================================

def predict_image(image: Image.Image):

    # STEP 1: Leaf validation
    leaf_result = validate_leaf(image)

    if not leaf_result["is_leaf"]:

        return {
            "status": "invalid_input",
            "disease": None,
            "confidence": None,
            "leaf_confidence": leaf_result["leaf_confidence"],
            "message": (
                "This image does not appear to contain a plant leaf. "
                "Please upload a clear image of a plant leaf."
            )
        }

    # STEP 2: Disease prediction
    disease_result = predict_disease(image)

    disease_result["leaf_confidence"] = (
        leaf_result["leaf_confidence"]
    )

    return disease_result
