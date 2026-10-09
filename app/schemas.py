# ==============================================================================
# PYDANTIC REQUEST AND RESPONSE SCHEMAS
# ------------------------------------------------------------------------------
# Purpose:
# Defines data validation contracts for incoming API requests and outgoing
# predictions. FastAPI uses these schemas to validate inputs, generate
# automatic OpenAPI/Swagger documentation, and serialize JSON responses.
#
# Key design decisions:
# 1. Exact raw features: The request model expects exactly the 19 raw features
#    from the customer record, excluding customerID, Churn, and engineered features.
# 2. Strict category enforcement: Uses typing.Literal to accept only valid
#    categories present during model training, rejecting unseen categories (HTTP 422).
# 3. Flexible missing values: Allows TotalCharges to be None (null), supporting
#    new customers whose lifetime spend is handled by the pipeline's median imputer.
# ==============================================================================

from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# Request Schema: Validates raw customer profile submitted for prediction
# ------------------------------------------------------------------------------
class ChurnPredictionRequest(BaseModel):
    """
    Input schema defining the 19 raw features required by the ML pipeline.
    All fields are validated before reaching the inference engine.
    """
    # Customer Demographics
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["No", "Yes"]
    Dependents: Literal["No", "Yes"]

    # Account & Contract Details
    tenure: float = Field(..., ge=0, description="Customer relationship duration in months (must be non-negative)")
    PhoneService: Literal["No", "Yes"]
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "No internet service", "Yes"]
    OnlineBackup: Literal["No", "No internet service", "Yes"]
    DeviceProtection: Literal["No", "No internet service", "Yes"]
    TechSupport: Literal["No", "No internet service", "Yes"]
    StreamingTV: Literal["No", "No internet service", "Yes"]
    StreamingMovies: Literal["No", "No internet service", "Yes"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["No", "Yes"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]

    # Billing & Financial Details
    MonthlyCharges: float = Field(..., ge=0, description="Current monthly bill (must be non-negative)")
    TotalCharges: Optional[float] = Field(default=None, ge=0, description="Lifetime spend (optional, handled by pipeline imputer)")


# ------------------------------------------------------------------------------
# Explainability Schemas: Individual feature contribution and grouped lists
# ------------------------------------------------------------------------------
class FeatureContribution(BaseModel):
    """
    Represents a single feature's directional contribution to the prediction logit.
    """
    feature: str = Field(..., description="Human-readable feature name (e.g. 'Contract: Month-to-month')")
    contribution: float = Field(..., description="Directional log-odds contribution score")


class ModelExplanation(BaseModel):
    """
    Container for top factors pushing toward vs away from churn.
    Derived mathematically from Logistic Regression: contribution = transformed_val * weight.
    """
    toward_churn: List[FeatureContribution] = Field(
        default_factory=list,
        description="Top factors associated with higher churn risk for this prediction",
    )
    away_from_churn: List[FeatureContribution] = Field(
        default_factory=list,
        description="Top factors associated with lower churn risk for this prediction",
    )


# ------------------------------------------------------------------------------
# Response Schema: Standardized payload returned by POST /predict
# ------------------------------------------------------------------------------
class ChurnPredictionResponse(BaseModel):
    """
    Output schema containing the predicted class, probability, model version,
    and optional model-derived feature contributions.
    """
    prediction: Union[int, str] = Field(
        ...,
        description="Predicted churn class (0 for Retained, 1 for Churn)",
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction probability of churn bounded between 0.0 and 1.0",
    )
    model_version: str = Field(
        ...,
        description="Semantic version of the model used for inference",
    )
    explanation: Optional[ModelExplanation] = Field(
        default=None,
        description="Model-based feature contributions for this specific prediction",
    )


