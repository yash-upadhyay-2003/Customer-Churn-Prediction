
import numpy as np
import pandas as pd

SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
AUTO_PAY = ["Bank transfer (automatic)", "Credit card (automatic)"]

RAW_FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
                "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]

def add_features(X):
    """Deterministic feature engineering (no learned parameters, so no leakage)."""
    X = X.copy()
    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # SeniorCitizen is a label, not a quantity: 0/1 -> No/Yes (kept categorical)
    X["SeniorCitizen"] = pd.to_numeric(X["SeniorCitizen"], errors="coerce").map({0: "No", 1: "Yes"})
    X["TenureYears"] = X["tenure"] / 12
    # tenure 0 or missing -> NaN (never invented); the pipeline imputer handles it
    X["AvgMonthlySpend"] = X["TotalCharges"] / X["tenure"].where(X["tenure"] > 0)
    X["ActiveServiceCount"] = (X[SERVICE_COLS] == "Yes").sum(axis=1)
    X["IsMonthToMonth"] = (X["Contract"] == "Month-to-month").astype(int)
    X["IsAutoPayment"] = X["PaymentMethod"].isin(AUTO_PAY).astype(int)
    return X
