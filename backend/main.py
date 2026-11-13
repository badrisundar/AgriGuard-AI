from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from pathlib import Path

from .database import get_db
from .models import Disease, ManagementGuideline, PredictionHistory
from .ml_model import predict_image


app = FastAPI(title="AgriGuard AI API")

# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "AgriGuard AI Backend is running"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))

        return {
            "status": "healthy",
            "database": "connected"
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    try:

        # ------------------------------------------
        # 1. Validate file type
        # ------------------------------------------

        allowed_types = [
            "image/jpeg",
            "image/png",
            "image/jpg",
            "image/webp"
        ]

        if file.content_type not in allowed_types:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": (
                        "Unsupported image format. "
                        "Please upload JPG, JPEG, PNG or WEBP."
                    )
                }
            )

        # ------------------------------------------
        # 2. Validate filename
        # ------------------------------------------

        filename = file.filename or "unknown_image"

        # Store only filename, not directory information
        safe_filename = Path(filename).name

        # ------------------------------------------
        # 3. Read uploaded image
        # ------------------------------------------

        contents = await file.read()

        if not contents:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "The uploaded file is empty."
                }
            )

        # ------------------------------------------
        # 4. Validate actual image content
        # ------------------------------------------

        try:
            image = Image.open(
                BytesIO(contents)
            ).convert("RGB")

        except UnidentifiedImageError:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "The uploaded file is not a valid image."
                }
            )

        # ------------------------------------------
        # 5. Run MobileNetV3 prediction
        # ------------------------------------------

        result = predict_image(image)

        if result["status"] == "invalid_input":
            history = PredictionHistory(
                disease_id=None,
                confidence=0,
                status="invalid_input",
                image_path=safe_filename
            )
            db.add(history)
            db.commit()

            return {
                "status": "invalid_input",
                "message": result["message"],
                "leaf_confidence": result["leaf_confidence"]
            }

        # ------------------------------------------
        # 6. Handle uncertain prediction
        # ------------------------------------------

        if result["status"] == "uncertain":

            history = PredictionHistory(
                disease_id=None,
                confidence=result["confidence"],
                status="uncertain",
                image_path=safe_filename
            )

            db.add(history)
            db.commit()

            return {
                "status": "uncertain",
                "disease": None,
                "confidence": result["confidence"],
                "filename": safe_filename,
                "message": (
                    "The model confidence is below the required "
                    "threshold. Please upload a clearer image "
                    "or consult an agricultural expert."
                )
            }

        if result["status"] == "unsupported_crop":
            history = PredictionHistory(
                disease_id=None,
                confidence=0.0000,
                status="unsupported_crop",
                image_path=safe_filename
            )
            
            db.add(history)
            db.commit()
            
            return {
                "status": "unsupported_crop",
                "message": result["message"],
                "leaf_confidence": result["leaf_confidence"],
                "crop_support_similarity": result["crop_support_similarity"]
            }

        # ------------------------------------------
        # 7. Handle healthy plant
        # ------------------------------------------
        
        if result["status"] == "healthy":
            history = PredictionHistory(
                disease_id=None,
                confidence=result["confidence"],
                status="healthy",
                image_path=safe_filename
        )
            
            db.add(history)
            db.commit()
            
            return {
                "status": "healthy",
                "disease": None,
                "confidence": result["confidence"],
                "predicted_class": result["predicted_class"],
                "crop": result["crop"],
                "leaf_confidence": result["leaf_confidence"],
                "filename": safe_filename,
                "message": result["message"]
            }

        # ------------------------------------------
        # 7. Get ML model class
        # ------------------------------------------

        model_class = result["disease"]

        # ------------------------------------------
        # 8. Find disease in PostgreSQL
        # ------------------------------------------

        disease = (
            db.query(Disease)
            .filter(
                Disease.model_class == model_class
            )
            .first()
        )

        if disease is None:

            db.rollback()

            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": (
                        "Disease information not found "
                        "in database."
                    ),
                    "model_class": model_class
                }
            )

        # ------------------------------------------
        # 9. Get management guidelines
        # ------------------------------------------

        guidelines = (
            db.query(ManagementGuideline)
            .filter(
                ManagementGuideline.disease_id == disease.id
            )
            .all()
        )

        # ------------------------------------------
        # 10. Save prediction history
        # ------------------------------------------

        history = PredictionHistory(
            disease_id=disease.id,
            confidence=result["confidence"],
            status="prediction",
            image_path=safe_filename
        )

        db.add(history)
        db.commit()

        # ------------------------------------------
        # 11. Prepare management information
        # ------------------------------------------

        management_data = []

        for guideline in guidelines:

            management_data.append({
                "type": guideline.management_type,
                "guideline": guideline.guideline,
                "source": guideline.source,
                "last_verified": (
                    str(guideline.last_verified)
                    if guideline.last_verified
                    else None
                )
            })

        # ------------------------------------------
        # 12. Final response
        # ------------------------------------------

        return {
            "status": "prediction",

            "prediction": {
                "disease": disease.disease_name,
                "crop": disease.crop,
                "scientific_name": disease.scientific_name,
                "confidence": result["confidence"]
            },

            "filename": safe_filename,

            "disease_information": {
                "description": disease.description,
                "symptoms": disease.symptoms,
                "prevention": disease.prevention
            },

            "management_guidelines": management_data,

            "disclaimer": (
                "This information is provided for decision support. "
                "AI detection is not a substitute for professional "
                "agricultural advice. Follow current local "
                "recommendations and product labels where applicable."
            )
        }

    # ----------------------------------------------
    # ERROR HANDLING
    # ----------------------------------------------

    except Exception as e:

        db.rollback()

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": (
                    "An error occurred while processing the image."
                ),
                "error": str(e)
            }
        )


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

@app.get("/history")
def get_prediction_history(
    db: Session = Depends(get_db)
):

    try:

        records = (
            db.query(
                PredictionHistory,
                Disease
            )
            .outerjoin(
                Disease,
                PredictionHistory.disease_id == Disease.id
            )
            .order_by(
                PredictionHistory.predicted_at.desc()
            )
            .all()
        )

        history_data = []

        for prediction, disease in records:

            history_data.append({
                "id": prediction.id,

                "filename": prediction.image_path,

                "crop": (
                    disease.crop
                    if disease
                    else None
                ),

                "disease": (
                    disease.disease_name
                    if disease
                    else None
                ),

                "confidence": float(
                    prediction.confidence
                ),

                "status": prediction.status,

                "predicted_at": (
                    str(prediction.predicted_at)
                    if prediction.predicted_at
                    else None
                )
            })

        return {
            "status": "success",
            "count": len(history_data),
            "history": history_data
        }

    except Exception as e:

        db.rollback()

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": (
                    "Unable to retrieve prediction history."
                ),
                "error": str(e)
            }
        )

# --------------------------------------------------
# GET ALL SUPPORTED DISEASES
# --------------------------------------------------

@app.get("/diseases")
def get_diseases(
    db: Session = Depends(get_db)
):

    try:

        diseases = (
            db.query(Disease)
            .order_by(
                Disease.crop,
                Disease.disease_name
            )
            .all()
        )

        disease_data = []

        for disease in diseases:

            disease_data.append({
                "id": disease.id,
                "crop": disease.crop,
                "disease": disease.disease_name,
                "scientific_name": disease.scientific_name,
                "model_class": disease.model_class
            })

        return {
            "status": "success",
            "count": len(disease_data),
            "diseases": disease_data
        }

    except Exception as e:

        db.rollback()

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Unable to retrieve diseases.",
                "error": str(e)
            }
        )