import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import pandas as pd

from app.config import settings
from app.schemas import ChurnPredictionRequest

RAW_FEATURE_NAMES = list(ChurnPredictionRequest.model_fields.keys())

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
    if raw_name.startswith("num__"):
        return NUM_MAP.get(raw_name[5:], raw_name[5:])
    elif raw_name.startswith("cat__"):
        name = raw_name[5:]
        for col in sorted(CAT_COLS, key=len, reverse=True):
            if name.startswith(col + "_"):
                val = name[len(col) + 1 :]
                return f"{CAT_LABEL_MAP.get(col, col)}: {val}"
    return raw_name


class ModelLoadingError(RuntimeError):
    pass


class PredictionError(RuntimeError):
    pass


class ChurnPredictor:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_version: Optional[str] = None,
        auto_load: bool = True,
    ) -> None:
        self.model_path: Path = Path(model_path) if model_path else settings.MODEL_PATH
        self.model_version: str = model_version or settings.MODEL_VERSION
        self._pipeline: Any = None
        self._positive_class_index: int = 1
        self._feature_names: Optional[List[str]] = None
        self._coefs: Any = None

        if auto_load:
            self.load_model()

    def load_model(self) -> None:
        if not self.model_path.exists():
            raise ModelLoadingError(
                f"Model artifact not found at configured path: {self.model_path.name}"
            )

        artifacts_dir = self.model_path.parent
        if str(artifacts_dir) not in sys.path:
            sys.path.insert(0, str(artifacts_dir))

        try:
            self._pipeline = joblib.load(self.model_path)
        except Exception as exc:
            raise ModelLoadingError(f"Failed to load model pipeline: {exc}") from exc

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

        if hasattr(self._pipeline, "named_steps"):
            preprocess_step = self._pipeline.named_steps.get("preprocess")
            model_step = self._pipeline.named_steps.get("model")
            if preprocess_step and hasattr(preprocess_step, "get_feature_names_out"):
                self._feature_names = preprocess_step.get_feature_names_out()
            if model_step and hasattr(model_step, "coef_"):
                self._coefs = model_step.coef_[0]

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            self.load_model()
        return self._pipeline

    def _calculate_explanation(self, df: pd.DataFrame, top_n: int = 3) -> Dict[str, Any]:
        if self._feature_names is None or self._coefs is None:
            return {"toward_churn": [], "away_from_churn": []}
        try:
            X_feat = self._pipeline.named_steps["features"].transform(df)
            X_trans = self._pipeline.named_steps["preprocess"].transform(X_feat)
            row_trans = X_trans[0]
            contributions = row_trans * self._coefs

            pos_items = sorted(
                [(f, c) for f, c in zip(self._feature_names, contributions) if c > 0],
                key=lambda x: x[1],
                reverse=True,
            )
            neg_items = sorted(
                [(f, c) for f, c in zip(self._feature_names, contributions) if c < 0],
                key=lambda x: x[1],
            )

            toward = []
            seen_toward = set()
            for f, c in pos_items:
                display = _format_feature_name(f)
                if display not in seen_toward:
                    seen_toward.add(display)
                    toward.append({"feature": display, "contribution": round(float(c), 4)})
                    if len(toward) >= top_n:
                        break

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
            return {"toward_churn": [], "away_from_churn": []}

    def predict(
        self,
        request: Union[ChurnPredictionRequest, Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self._pipeline is None:
            self.load_model()

        try:
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

            df = pd.DataFrame([data])[RAW_FEATURE_NAMES]

            raw_pred = self._pipeline.predict(df)[0]
            raw_proba = self._pipeline.predict_proba(df)[0]

            prediction = int(raw_pred)
            probability = float(raw_proba[self._positive_class_index])
            explanation = self._calculate_explanation(df, top_n=3)

            return {
                "prediction": prediction,
                "probability": probability,
                "model_version": self.model_version,
                "explanation": explanation,
            }
        except Exception as exc:
            raise PredictionError(f"Error during model prediction: {exc}") from exc


predictor = ChurnPredictor()


def get_predictor() -> ChurnPredictor:
    return predictor
