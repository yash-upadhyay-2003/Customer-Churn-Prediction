"""
FastAPI Backend Application for Customer Churn Prediction.

Role in Architecture:
- Exposes RESTful HTTP endpoints connecting web clients (like the Streamlit frontend)
  to the machine learning inference engine.
- Automatically validates incoming JSON request payloads against Pydantic schemas.
- Manages Cross-Origin Resource Sharing (CORS) permissions for web browsers.
- Implements production health monitoring via /health.
- Coordinates inference execution, explainability computation, and error response formatting.
- Auto-generates interactive OpenAPI documentation at http://localhost:8000/docs.
"""

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

# Initialize application logger for diagnostic and audit messages
logger = logging.getLogger("churn_api")

# -----------------------------------------------------------------------------
# 1. APPLICATION INITIALIZATION
# -----------------------------------------------------------------------------
# Initialize the FastAPI app with metadata configured in app/config.py.
# FastAPI automatically generates interactive documentation at /docs (Swagger UI)
# and /redoc (ReDoc) using these parameters.
app = FastAPI(
    title=settings.API_TITLE,
    description="Production API for Customer Churn Prediction using Logistic Regression pipeline",
    version=settings.MODEL_VERSION,
)

# -----------------------------------------------------------------------------
# 2. CORS (CROSS-ORIGIN RESOURCE SHARING) CONFIGURATION
# -----------------------------------------------------------------------------
# Web browsers enforce the Same-Origin Policy for security.
# Because the Streamlit frontend runs on port 8501 and FastAPI runs on port 8000,
# the browser treats requests between them as cross-origin.
# This middleware explicitly permits requests originating from the Streamlit UI.
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


# -----------------------------------------------------------------------------
# 3. ROOT DISCOVERABILITY ENDPOINT
# -----------------------------------------------------------------------------
@app.get("/", tags=["General"])
def read_root() -> Dict[str, str]:
    """
    Root discoverability endpoint.

    Provides basic service information and links to interactive documentation
    and system health checks for developers and automated monitors.
    """
    return {
        "title": settings.API_TITLE,
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
    }


# -----------------------------------------------------------------------------
# 4. SYSTEM HEALTH & READINESS PROBE
# -----------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
def health_check(
    predictor: ChurnPredictor = Depends(get_predictor),
) -> Dict[str, str]:
    """
    Production health check and model readiness probe.

    Purpose in Production:
    - Cloud platforms (Render, AWS, Kubernetes) poll this endpoint to verify
      that the service is ready to accept user traffic.
    - Uses FastAPI dependency injection (Depends) to check that the ML model
      pipeline is loaded in memory and functional.

    Responses:
    - 200 OK: Service is healthy and model pipeline is loaded in memory.
    - 503 SERVICE UNAVAILABLE: Model artifact could not be loaded into memory.
    - 500 INTERNAL SERVER ERROR: Unexpected server failure.
    """
    try:
        # Verify that the underlying scikit-learn pipeline is resident in memory
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


# -----------------------------------------------------------------------------
# 5. PREDICTION & EXPLAINABILITY ENDPOINT
# -----------------------------------------------------------------------------
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
    """
    End-to-end churn prediction and factor explanation endpoint.

    Workflow:
    1. Input Validation: FastAPI automatically validates the JSON body against
       ChurnPredictionRequest. Invalid data immediately returns HTTP 422 Unprocessable Entity.
    2. Dependency Injection: The pre-loaded ChurnPredictor singleton is injected.
    3. Prediction & Explanation: predictor.predict() runs feature engineering,
       scaling/encoding, classification, probability scoring, and log-odds contributions.
    4. Response Validation: Output is serialized and validated against ChurnPredictionResponse.
    5. Error Handling: Custom exceptions are caught and mapped to standard HTTP error codes
       without leaking internal tracebacks or filesystem paths.

    Args:
        request: Validated customer feature data matching ChurnPredictionRequest.
        predictor: Injected ChurnPredictor singleton instance.

    Returns:
        ChurnPredictionResponse containing prediction, probability, version, and explanations.
    """
    try:
        # Execute model pipeline inference and feature explanation math
        result = predictor.predict(request)
        # Validate output dictionary against Pydantic response contract
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

