
# ==============================================================================
# PIPELINE FEATURE ENGINEERING MODULE
# ------------------------------------------------------------------------------
# This module defines the custom feature engineering function bundled into the
# trained scikit-learn pipeline (churn_pipeline.joblib) via FunctionTransformer.
#
# Role in the architecture:
# Raw Input (19 features) -> add_features() -> 24 features -> ColumnTransformer
#
# Because all calculations are deterministic (mathematical formulas with no
# learned weights/parameters), it executes safely during inference without
# risking data leakage.
# ==============================================================================

import numpy as np
import pandas as pd

# List of optional internet and support services used to calculate active subscription count.
SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]

# Payment methods classified as automatic (bank transfer or recurring credit card).
AUTO_PAY = ["Bank transfer (automatic)", "Credit card (automatic)"]

# The 19 raw features collected from the user and expected by the model contract.
RAW_FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
                "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]


def add_features(X):
    """
    Deterministic feature engineering step applied before imputation and scaling.

    Input:
        X (pd.DataFrame): DataFrame containing raw customer features.

    Output:
        X (pd.DataFrame): DataFrame augmented with 5 new engineered features:
            1. TenureYears: Tenure converted from months to years.
            2. AvgMonthlySpend: Historical average spend per month.
            3. ActiveServiceCount: Count of optional services enabled.
            4. IsMonthToMonth: Binary flag indicating high-churn flexible contract.
            5. IsAutoPayment: Binary flag indicating automatic payment method.
    """
    # Create an explicit copy to avoid modifying the caller's DataFrame in place
    X = X.copy()

    # Ensure numerical columns are properly parsed as float/numeric, converting invalid text to NaN
    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # SeniorCitizen is a demographic flag (0 or 1), not a continuous measurement.
    # We map 0 -> 'No' and 1 -> 'Yes' so the OneHotEncoder treats it as a standard categorical variable.
    X["SeniorCitizen"] = pd.to_numeric(X["SeniorCitizen"], errors="coerce").map({0: "No", 1: "Yes"})

    # Feature 1: Convert tenure from months into years for linear relationship modeling
    X["TenureYears"] = X["tenure"] / 12

    # Feature 2: Average historical spend per active month.
    # When tenure is 0 (brand new customer), we leave it as NaN to prevent division by zero;
    # the downstream SimpleImputer in the pipeline will cleanly fill missing values with the median.
    X["AvgMonthlySpend"] = X["TotalCharges"] / X["tenure"].where(X["tenure"] > 0)

    # Feature 3: Engagement metric - sum of all active digital services (0 to 6)
    X["ActiveServiceCount"] = (X[SERVICE_COLS] == "Yes").sum(axis=1)

    # Feature 4: High-risk indicator - month-to-month contracts have the highest churn propensity
    X["IsMonthToMonth"] = (X["Contract"] == "Month-to-month").astype(int)

    # Feature 5: Retention indicator - automatic payments typically correlate with higher retention
    X["IsAutoPayment"] = X["PaymentMethod"].isin(AUTO_PAY).astype(int)

    return X

