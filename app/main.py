import logging
from typing import Dict

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.predictor import (
    ChurnPredictor,
    ModelLoadingError,
    PredictionError,
    get_predictor,
)
from app.schemas import ChurnPredictionRequest, ChurnPredictionResponse

logger = logging.getLogger("churn_api")

app = FastAPI(
    title=settings.API_TITLE,
    description="Production API for Customer Churn Prediction using Logistic Regression pipeline",
    version=settings.MODEL_VERSION,
)

ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def read_root() -> Dict[str, str]:
    return {
        "title": settings.API_TITLE,
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", tags=["Monitoring"])
def health_check(
    predictor: ChurnPredictor = Depends(get_predictor),
) -> Dict[str, str]:
    try:
        if predictor.pipeline is None:
            raise ModelLoadingError("Pipeline object is unavailable")
        return {
            "status": "healthy",
            "model_version": settings.MODEL_VERSION,
        }
    except ModelLoadingError as exc:
        logger.error(f"Health check failed - model unavailable: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable",
        )
    except Exception as exc:
        logger.error(f"Health check failed - unexpected error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed",
        )


@app.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
def predict_churn(
    request: ChurnPredictionRequest,
    predictor: ChurnPredictor = Depends(get_predictor),
) -> ChurnPredictionResponse:
    try:
        result = predictor.predict(request)
        return ChurnPredictionResponse(**result)
    except ModelLoadingError as exc:
        logger.error(f"Model loading failure during prediction: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable for prediction",
        )
    except PredictionError as exc:
        logger.error(f"Prediction inference failure: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference execution failed",
        )
    except Exception as exc:
        logger.error(f"Unexpected error during prediction: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction",
        )
