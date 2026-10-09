from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field


class ChurnPredictionRequest(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["No", "Yes"]
    Dependents: Literal["No", "Yes"]
    tenure: float = Field(..., ge=0)
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
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: Optional[float] = Field(default=None, ge=0)


class FeatureContribution(BaseModel):
    feature: str = Field(..., description="Human-readable feature name")
    contribution: float = Field(..., description="Directional log-odds contribution score")


class ModelExplanation(BaseModel):
    toward_churn: List[FeatureContribution] = Field(
        default_factory=list,
        description="Top factors associated with higher churn risk for this prediction",
    )
    away_from_churn: List[FeatureContribution] = Field(
        default_factory=list,
        description="Top factors associated with lower churn risk for this prediction",
    )


class ChurnPredictionResponse(BaseModel):
    prediction: Union[int, str] = Field(
        ...,
        description="Predicted churn class (0 or 1, or class label)",
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction probability of churn",
    )
    model_version: str = Field(
        ...,
        description="Model version identifier",
    )
    explanation: Optional[ModelExplanation] = Field(
        default=None,
        description="Model-based feature contributions for this specific prediction",
    )

