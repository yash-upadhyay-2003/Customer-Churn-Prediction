"""
Inference and Explainability Module for Customer Churn Prediction.

Role in Architecture:
- Acts as the bridge between the FastAPI web framework and the serialized ML model.
- Loads the trained scikit-learn Pipeline (churn_pipeline.joblib) and inspects its internals.
- Formats single-customer input dictionaries into 1-row pandas DataFrames matching training schema.
- Runs inference (.predict() and .predict_proba()) to determine churn classification.
- Calculates exact mathematical feature contributions using Logistic Regression log-odds (z_i * beta_i).
- Formats technical feature names into business-friendly labels for web UI presentation.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import pandas as pd

from app.config import settings
from app.schemas import ChurnPredictionRequest

# -----------------------------------------------------------------------------
# 1. RAW FEATURE COLUMN ORDER
# -----------------------------------------------------------------------------
# Extract the exact feature field names expected by the model from the Pydantic schema.
# Enforcing this exact column order in pandas DataFrames prevents input mismatch errors.
RAW_FEATURE_NAMES = list(ChurnPredictionRequest.model_fields.keys())

# -----------------------------------------------------------------------------
# 2. FEATURE NAME MAPPING FOR HUMAN-READABLE EXPLANATIONS
# -----------------------------------------------------------------------------
# Scikit-learn's OneHotEncoder produces technical column names such as:
#   "cat__InternetService_Fiber optic" or "num__tenure"
# These lookup dictionaries map technical pipeline names into clean, readable labels
# suitable for non-technical business users viewing the Streamlit dashboard.

# List of raw categorical columns before one-hot encoding
CAT_COLS = [
    "PaymentMethod",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "MultipleLines",
    "PhoneService",
    "PaperlessBilling",
    "SeniorCitizen",
    "Contract",
    "Partner",
    "Dependents",
    "gender",
]

# Business-friendly display names for categorical feature categories
CAT_LABEL_MAP = {
    "PaymentMethod": "Payment Method",
    "InternetService": "Internet Service",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "MultipleLines": "Multiple Lines",
    "PhoneService": "Phone Service",
    "PaperlessBilling": "Paperless Billing",
    "SeniorCitizen": "Senior Citizen",
    "Contract": "Contract",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "gender": "Gender",
}

# Mapping for numeric and engineered feature names
# Note: 'tenure' and 'TenureYears' both map to 'Tenure' for consistent display
NUM_MAP = {
    "tenure": "Tenure",
    "TenureYears": "Tenure",
    "MonthlyCharges": "Monthly Charges",
    "TotalCharges": "Total Charges",
    "AvgMonthlySpend": "Average Monthly Spend",
    "ActiveServiceCount": "Active Services Count",
    "IsMonthToMonth": "Contract: Month-to-month",
    "IsAutoPayment": "Automatic Payment",
}


def _format_feature_name(raw_name: str) -> str:
    """
    Format a technical pipeline feature name into a clean, human-readable display string.

    Examples:
        - "num__tenure" -> "Tenure"
        - "cat__Contract_Month-to-month" -> "Contract: Month-to-month"
        - "cat__InternetService_Fiber optic" -> "Internet Service: Fiber optic"

    Args:
        raw_name: Technical feature name output by ColumnTransformer.

    Returns:
        Human-readable feature name string.
    """
    # Handle numeric columns prefixed with "num__"
    if raw_name.startswith("num__"):
        return NUM_MAP.get(raw_name[5:], raw_name[5:])
    # Handle one-hot encoded categorical columns prefixed with "cat__"
    elif raw_name.startswith("cat__"):
        name = raw_name[5:]
        # Check against known categorical column prefixes sorted longest first to avoid substring collision
        for col in sorted(CAT_COLS, key=len, reverse=True):
            if name.startswith(col + "_"):
                val = name[len(col) + 1 :]
                return f"{CAT_LABEL_MAP.get(col, col)}: {val}"
    return raw_name


# -----------------------------------------------------------------------------
# 3. DOMAIN-SPECIFIC CUSTOM EXCEPTIONS
# -----------------------------------------------------------------------------
# Dedicated exceptions allow the FastAPI layer to distinguish between startup/loading
# failures (HTTP 503) and runtime prediction failures (HTTP 500).

class ModelLoadingError(RuntimeError):
    """Raised when the serialized model file cannot be found or loaded into memory."""
    pass


class PredictionError(RuntimeError):
    """Raised when inference or feature preprocessing fails on input data."""
    pass


# -----------------------------------------------------------------------------
# 4. CHURN PREDICTOR CLASS
# -----------------------------------------------------------------------------
class ChurnPredictor:
    """
    Production inference engine encapsulating the scikit-learn Pipeline.

    Responsibilities:
    1. Deserializing 'churn_pipeline.joblib' from disk into memory.
    2. Dynamically inspecting positive class index (1 or 'Yes').
    3. Extracting transformed feature names and Logistic Regression coefficients.
    4. Converting input customer records into single-row DataFrames.
    5. Executing end-to-end inference (feature engineering + preprocessing + classification).
    6. Computing mathematical feature contributions (z_i * beta_i) for explainability.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_version: Optional[str] = None,
        auto_load: bool = True,
    ) -> None:
        """
        Initialize the predictor with artifact path and version configuration.

        Args:
            model_path: Path to .joblib file (defaults to settings.MODEL_PATH).
            model_version: Semantic version string (defaults to settings.MODEL_VERSION).
            auto_load: If True, loads the model immediately during startup.
        """
        self.model_path: Path = Path(model_path) if model_path else settings.MODEL_PATH
        self.model_version: str = model_version or settings.MODEL_VERSION
        self._pipeline: Any = None
        self._positive_class_index: int = 1
        self._feature_names: Optional[List[str]] = None
        self._coefs: Any = None

        if auto_load:
            self.load_model()

    def load_model(self) -> None:
        """
        Load the serialized scikit-learn pipeline from disk and extract explainability metadata.

        Important Steps:
        - Verifies artifact existence before attempting to read.
        - Appends artifacts/ to sys.path so joblib can unpickle custom functions (e.g. churn_features.add_features).
        - Identifies index of churn class (1) in pipeline.classes_ for predict_proba indexing.
        - Caches feature names and model coefficients for rapid explainability calculations.
        """
        # Step 1: Ensure the artifact file actually exists on the filesystem
        if not self.model_path.exists():
            raise ModelLoadingError(
                f"Model artifact not found at configured path: {self.model_path.name}"
            )

        # Step 2: Ensure the artifacts directory is on Python's module search path
        # This is critical so joblib can find 'churn_features.py' referenced inside the pickled FunctionTransformer
        artifacts_dir = self.model_path.parent
        if str(artifacts_dir) not in sys.path:
            sys.path.insert(0, str(artifacts_dir))

        # Step 3: Deserialize the complete pipeline using joblib
        try:
            self._pipeline = joblib.load(self.model_path)
        except Exception as exc:
            raise ModelLoadingError(f"Failed to load model pipeline: {exc}") from exc

        # Step 4: Resolve the positive class index (churn = 1) from the trained model
        classes = getattr(self._pipeline, "classes_", None)
        if (
            classes is None
            and hasattr(self._pipeline, "named_steps")
            and "model" in self._pipeline.named_steps
        ):
            classes = getattr(self._pipeline.named_steps["model"], "classes_", None)

        if classes is not None:
            classes_list = list(classes)
            if 1 in classes_list:
                self._positive_class_index = classes_list.index(1)
            elif "Yes" in classes_list:
                self._positive_class_index = classes_list.index("Yes")
            elif len(classes_list) > 1:
                self._positive_class_index = 1
            else:
                self._positive_class_index = 0

        # Step 5: Extract feature names and Logistic Regression coefficients for explainability
        if hasattr(self._pipeline, "named_steps"):
            preprocess_step = self._pipeline.named_steps.get("preprocess")
            model_step = self._pipeline.named_steps.get("model")
            # get_feature_names_out() returns the names of all 51 transformed columns
            if preprocess_step and hasattr(preprocess_step, "get_feature_names_out"):
                self._feature_names = preprocess_step.get_feature_names_out()
            # coef_[0] contains the 51 logistic regression weights (betas)
            if model_step and hasattr(model_step, "coef_"):
                self._coefs = model_step.coef_[0]

    @property
    def pipeline(self) -> Any:
        """Access the underlying loaded pipeline, lazy-loading if necessary."""
        if self._pipeline is None:
            self.load_model()
        return self._pipeline

    def _calculate_explanation(self, df: pd.DataFrame, top_n: int = 3) -> Dict[str, Any]:
        """
        Calculate model-based feature contributions using Logistic Regression log-odds math.

        Mathematical Basis:
        In Logistic Regression:
            logit(p) = beta_0 + sum(z_i * beta_i)
        Where:
            z_i = transformed feature value for this specific customer (after scaling/one-hot encoding)
            beta_i = trained model coefficient for feature i
            c_i = z_i * beta_i is the exact log-odds contribution of feature i

        Interpretation:
        - If c_i > 0: This feature pushes the logit higher -> increases probability of Churn.
        - If c_i < 0: This feature pushes the logit lower -> decreases probability of Churn.

        Deduplication:
        - Related features sharing the same label (e.g. tenure and TenureYears both showing as "Tenure")
          are deduplicated so a diverse set of distinct business factors is presented.

        Args:
            df: Single-row DataFrame containing customer input.
            top_n: Number of top driving factors to return (default 3).

        Returns:
            Dictionary containing 'toward_churn' and 'away_from_churn' factor lists.
        """
        if self._feature_names is None or self._coefs is None:
            return {"toward_churn": [], "away_from_churn": []}
        try:
            # Transform customer input through feature engineering step
            X_feat = self._pipeline.named_steps["features"].transform(df)
            # Transform engineered dataframe through preprocessing step (scaling & encoding)
            X_trans = self._pipeline.named_steps["preprocess"].transform(X_feat)
            row_trans = X_trans[0]
            # Multiply customer's transformed feature values by the model's coefficients
            contributions = row_trans * self._coefs

            # Positive contributions push TOWARD churn (sorted highest positive first)
            pos_items = sorted(
                [(f, c) for f, c in zip(self._feature_names, contributions) if c > 0],
                key=lambda x: x[1],
                reverse=True,
            )
            # Negative contributions push AWAY FROM churn (sorted strongest negative first)
            neg_items = sorted(
                [(f, c) for f, c in zip(self._feature_names, contributions) if c < 0],
                key=lambda x: x[1],
            )

            # Select top positive factors with deduplication
            toward = []
            seen_toward = set()
            for f, c in pos_items:
                display = _format_feature_name(f)
                if display not in seen_toward:
                    seen_toward.add(display)
                    toward.append({"feature": display, "contribution": round(float(c), 4)})
                    if len(toward) >= top_n:
                        break

            # Select top negative factors with deduplication
            away = []
            seen_away = set()
            for f, c in neg_items:
                display = _format_feature_name(f)
                if display not in seen_away and display not in seen_toward:
                    seen_away.add(display)
                    away.append({"feature": display, "contribution": round(float(c), 4)})
                    if len(away) >= top_n:
                        break

            return {"toward_churn": toward, "away_from_churn": away}
        except Exception:
            # Graceful fallback: return empty explanation lists if calculation fails
            return {"toward_churn": [], "away_from_churn": []}

    def predict(
        self,
        request: Union[ChurnPredictionRequest, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute end-to-end churn prediction for a single customer.

        Pipeline Execution:
        1. Validated customer dictionary is converted to a 1-row pandas DataFrame.
        2. Pipeline applies Feature Engineering (add_features): computes 5 derived features.
        3. Pipeline applies ColumnTransformer (preprocess): scales numerics, encodes categoricals.
        4. Pipeline applies LogisticRegression (model): computes predicted class and probability.
        5. Explainability engine computes top driving factors.

        Args:
            request: ChurnPredictionRequest Pydantic instance or dictionary of feature values.

        Returns:
            Dictionary matching ChurnPredictionResponse schema:
                - prediction: 0 (Stay) or 1 (Churn)
                - probability: float between 0.0 and 1.0 (churn probability)
                - model_version: semantic version string
                - explanation: top positive and negative contributing features

        Raises:
            PredictionError: If inference fails during execution.
        """
        # Ensure model pipeline is loaded
        if self._pipeline is None:
            self.load_model()

        try:
            # Standardize request into a clean dictionary
            if isinstance(request, ChurnPredictionRequest):
                data = request.model_dump()
            elif isinstance(request, dict):
                data = {k: request[k] for k in RAW_FEATURE_NAMES if k in request}
            else:
                data = (
                    request.model_dump()
                    if hasattr(request, "model_dump")
                    else dict(request)
                )

            # Construct 1-row DataFrame preserving exact column names and order
            df = pd.DataFrame([data])[RAW_FEATURE_NAMES]

            # Execute pipeline prediction (0 = Non-churn, 1 = Churn)
            raw_pred = self._pipeline.predict(df)[0]
            # Execute pipeline probability estimation ([P(No Churn), P(Churn)])
            raw_proba = self._pipeline.predict_proba(df)[0]

            prediction = int(raw_pred)
            probability = float(raw_proba[self._positive_class_index])
            # Generate feature contribution explanation
            explanation = self._calculate_explanation(df, top_n=3)

            return {
                "prediction": prediction,
                "probability": probability,
                "model_version": self.model_version,
                "explanation": explanation,
            }
        except Exception as exc:
            raise PredictionError(f"Error during model prediction: {exc}") from exc


# -----------------------------------------------------------------------------
# 5. SINGLETON INSTANCE AND DEPENDENCY INJECTION HELPER
# -----------------------------------------------------------------------------
# Reusing a single predictor instance across API requests prevents reloading the
# model file from disk on every HTTP call, ensuring fast sub-50ms response times.
predictor = ChurnPredictor()


def get_predictor() -> ChurnPredictor:
    """Dependency injection provider function used by FastAPI route handlers."""
    return predictor

