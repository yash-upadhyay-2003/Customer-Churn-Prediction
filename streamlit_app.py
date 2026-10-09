"""
Streamlit Web Frontend Application for Customer Churn Prediction.

Role in Architecture:
- Provides a clean, modern, interactive web dashboard for business stakeholders.
- Decoupled Presentation Layer: Contains ZERO scikit-learn or joblib imports.
  All validation, preprocessing, and inference are delegated to the FastAPI backend.
- Communicates with FastAPI over HTTP:
    - GET  /health  -> sidebar service status monitoring.
    - POST /predict -> customer feature submission and prediction retrieval.
- Displays prediction results: predicted class, probability score, visual risk bands,
  driving factor contributions, and non-causal statistical disclaimers.
"""

import os
from typing import Any, Dict, List, Optional
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# 1. API CONFIGURATION & ENVIRONMENT LOOKUP
# -----------------------------------------------------------------------------
# Sourced from environment variable API_BASE_URL (useful when deploying to Streamlit Cloud
# pointing to a remote FastAPI URL on Render). Defaults to local FastAPI dev server.
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def main() -> None:
    """
    Main Streamlit application entry point.

    Coordinates UI rendering, user input collection, client-side validation,
    API communication, and visualization of churn predictions and factor explanations.
    """
    # Configure browser tab title and responsive layout
    st.set_page_config(
        page_title="Customer Churn Prediction",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # -------------------------------------------------------------------------
    # 2. CUSTOM CSS STYLING & DESIGN SYSTEM
    # -------------------------------------------------------------------------
    # Injects clean, production-grade SaaS styles: neutral typography, subtle borders,
    # and structured card layouts to create a polished executive assessment interface.
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3.5rem;
            max-width: 1040px;
        }
        .app-title {
            font-size: 1.85rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.25rem;
            letter-spacing: -0.02em;
        }
        .app-desc {
            font-size: 0.95rem;
            color: #475569;
            margin-bottom: 1.75rem;
            line-height: 1.5;
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: #1e293b;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid #e2e8f0;
            margin-top: 1.25rem;
            margin-bottom: 0.85rem;
        }
        .result-box {
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 1.5rem;
            background-color: #ffffff;
            margin-top: 2rem;
            margin-bottom: 1.5rem;
        }
        .result-header {
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
        }
        .result-status {
            font-size: 1.75rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            line-height: 1.2;
            margin-bottom: 0.25rem;
        }
        .result-substatus {
            font-size: 0.875rem;
            color: #475569;
            margin-bottom: 1.25rem;
        }
        .stat-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 1rem 1.15rem;
            height: 100%;
        }
        .stat-label {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-bottom: 0.3rem;
        }
        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #0f172a;
        }
        .scale-markers {
            display: flex;
            justify-content: space-between;
            font-size: 0.7rem;
            color: #94a3b8;
            margin-top: 0.35rem;
        }
        .factor-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 1rem 1.15rem;
            margin-top: 0.5rem;
            height: 100%;
        }
        .factor-heading {
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.6rem;
        }
        .factor-item {
            font-size: 0.875rem;
            color: #334155;
            padding: 0.25rem 0;
            line-height: 1.4;
        }
        .info-panel {
            background-color: #f8fafc;
            border-left: 3px solid #0284c7;
            padding: 0.85rem 1rem;
            border-radius: 0 4px 4px 0;
            font-size: 0.9rem;
            color: #334155;
            line-height: 1.5;
            margin-top: 1.25rem;
        }
        .meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-top: 1.25rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
            font-size: 0.8rem;
            color: #64748b;
        }
        .meta-cell strong {
            color: #334155;
        }
        .disclaimer-text {
            font-size: 0.78rem;
            color: #64748b;
            margin-top: 0.6rem;
            line-height: 1.4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 3. PAGE HEADER
    # -------------------------------------------------------------------------
    st.markdown('<div class="app-title">Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-desc">'
        "Estimate customer churn risk using the deployed Logistic Regression pipeline. "
        "Enter customer demographic, billing, and subscribed service information below "
        "to generate a prediction with model-based factor contributions."
        "</div>",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 4. SIDEBAR: HEALTH CHECK & RISK REFERENCE
    # -------------------------------------------------------------------------
    # Provides users with a live connectivity test to the backend API without
    # needing to submit customer prediction data.
    with st.sidebar:
        st.subheader("Service Status")
        if st.button("Check Service Status"):
            try:
                res = requests.get(f"{API_BASE_URL}/health", timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"API is active (Model version: {data.get('model_version', '1.0.0')})")
                else:
                    st.warning(f"Service returned status HTTP {res.status_code}")
            except Exception:
                st.error("Service is currently unreachable.")

        st.markdown("---")
        st.markdown(
            """
            **Prediction Risk Level Bands**
            - Low Risk: &lt; 30.0%
            - Moderate Risk: 30.0% – 59.9%
            - High Risk: &ge; 60.0%
            
            *Presentation bands are based on model-estimated churn probability.*
            """
        )

    # -------------------------------------------------------------------------
    # 5. INPUT FORM SECTION 1: CUSTOMER PROFILE (DEMOGRAPHICS)
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-title">1. Customer Profile</div>', unsafe_allow_html=True)
    cp1, cp2, cp3, cp4 = st.columns(4)
    with cp1:
        gender = st.selectbox("Gender", options=["Female", "Male"])
    with cp2:
        senior_citizen_opt = st.selectbox(
            "Senior Citizen",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
    with cp3:
        partner = st.selectbox("Partner", options=["No", "Yes"])
    with cp4:
        dependents = st.selectbox("Dependents", options=["No", "Yes"])


    # -------------------------------------------------------------------------
    # 6. INPUT FORM SECTION 2: ACCOUNT & BILLING DETAILS
    # -------------------------------------------------------------------------
    # Captures contractual duration and payment terms.
    # Note: 'TotalCharges' is optional. If left blank, it defaults to None and is
    # safely handled via median imputation in the ML preprocessing pipeline.
    st.markdown('<div class="section-title">2. Account Details</div>', unsafe_allow_html=True)
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0.0,
            value=12.0,
            step=1.0,
            help="Total active customer relationship duration in months.",
        )
        contract = st.selectbox(
            "Contract Term",
            options=["Month-to-month", "One year", "Two year"],
        )
    with ac2:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=65.0,
            step=0.5,
            help="Customer current monthly charge amount.",
        )
        paperless_billing = st.selectbox("Paperless Billing", options=["No", "Yes"])
    with ac3:
        payment_method = st.selectbox(
            "Payment Method",
            options=[
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ],
        )
        total_charges_raw = st.text_input(
            "Total Charges ($) - Optional",
            value="",
            help="Accumulated lifetime charges. Leave blank for new accounts (handled by pipeline imputation).",
        )

    # -------------------------------------------------------------------------
    # 7. INPUT FORM SECTION 3: SUBSCRIBED SERVICES
    # -------------------------------------------------------------------------
    # Captures phone, internet, streaming, and cybersecurity add-ons.
    # These services are aggregated by feature engineering into 'ActiveServiceCount'.
    st.markdown('<div class="section-title">3. Services</div>', unsafe_allow_html=True)
    sv1, sv2, sv3 = st.columns(3)
    with sv1:
        phone_service = st.selectbox("Phone Service", options=["No", "Yes"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            options=["No", "No phone service", "Yes"],
        )
        internet_service = st.selectbox(
            "Internet Service",
            options=["DSL", "Fiber optic", "No"],
        )
    with sv2:
        online_security = st.selectbox(
            "Online Security",
            options=["No", "No internet service", "Yes"],
        )
        online_backup = st.selectbox(
            "Online Backup",
            options=["No", "No internet service", "Yes"],
        )
        device_protection = st.selectbox(
            "Device Protection",
            options=["No", "No internet service", "Yes"],
        )
    with sv3:
        tech_support = st.selectbox(
            "Tech Support",
            options=["No", "No internet service", "Yes"],
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            options=["No", "No internet service", "Yes"],
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            options=["No", "No internet service", "Yes"],
        )

    st.write("")
    predict_clicked = st.button("Predict Churn", type="primary", use_container_width=True)

    # -------------------------------------------------------------------------
    # 8. PREDICTION TRIGGER & CLIENT-SIDE VALIDATION
    # -------------------------------------------------------------------------
    if predict_clicked:
        # Validate TotalCharges input: convert to float or set None for blank inputs
        total_charges_parsed: Optional[float] = None
        cleaned_total = total_charges_raw.strip()
        if cleaned_total:
            try:
                parsed_val = float(cleaned_total)
                if parsed_val < 0:
                    st.error("Total Charges must not be negative.")
                    return
                total_charges_parsed = parsed_val
            except ValueError:
                st.error("Total Charges must be a valid number or left blank.")
                return

        # Construct JSON request payload matching ChurnPredictionRequest schema
        payload: Dict[str, Any] = {
            "gender": gender,
            "SeniorCitizen": senior_citizen_opt,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": float(tenure),
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": total_charges_parsed,
        }

        # ---------------------------------------------------------------------
        # 9. HTTP API CALL TO FASTAPI BACKEND
        # ---------------------------------------------------------------------
        with st.spinner("Requesting prediction from ML service..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/predict",
                    json=payload,
                    timeout=10,
                )
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Unable to connect to the prediction API at {API_BASE_URL}. "
                    "Please ensure the FastAPI service is running."
                )
                return
            except requests.exceptions.Timeout:
                st.error("The prediction request timed out. Please try again.")
                return
            except requests.exceptions.RequestException:
                st.error("An error occurred while contacting the prediction service.")
                return

        # ---------------------------------------------------------------------
        # 10. PREDICTION RESPONSE PARSING & METRIC EXTRACTION
        # ---------------------------------------------------------------------
        if response.status_code == 200:
            try:
                result = response.json()
                raw_prediction = result.get("prediction")
                probability = float(result.get("probability", 0.0))
                model_version = str(result.get("model_version", "1.0.0"))
                explanation = result.get("explanation") or {}
                toward_churn: List[Dict[str, Any]] = explanation.get("toward_churn", [])
                away_from_churn: List[Dict[str, Any]] = explanation.get("away_from_churn", [])

                # 1. Determine Predicted Status & Class Presentation
                is_churn = (raw_prediction == 1) or (str(raw_prediction).lower() in ["1", "yes", "churn"])
                if is_churn:
                    status_text = "LIKELY TO CHURN"
                    status_subtext = "Predicted Class: Churn (1)"
                    status_color = "#991b1b"
                else:
                    status_text = "LIKELY TO STAY"
                    status_subtext = "Predicted Class: Retained (0)"
                    status_color = "#166534"

                # 2. Risk Level Presentation Bands based on model probability
                if probability < 0.30:
                    risk_level = "Low Risk"
                    risk_color = "#166534"
                    meaning_text = (
                        "The model estimates a relatively low probability that this customer will churn "
                        "based on the information provided."
                    )
                elif probability < 0.60:
                    risk_level = "Moderate Risk"
                    risk_color = "#d97706"
                    meaning_text = (
                        "The model estimates a moderate churn probability. "
                        "This customer may warrant closer attention."
                    )
                else:
                    risk_level = "High Risk"
                    risk_color = "#dc2626"
                    meaning_text = (
                        "The model estimates a relatively high churn probability "
                        "based on the information provided."
                    )

                prob_percent = f"{probability * 100:.2f}%"

                # -------------------------------------------------------------
                # 11. UI RENDERING: ASSESSMENT BANNER & METRICS
                # -------------------------------------------------------------
                # Render primary Customer Churn Assessment Banner
                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="result-header" style="color: {status_color};">Customer Churn Assessment</div>
                        <div class="result-status" style="color: {status_color};">{status_text}</div>
                        <div class="result-substatus">{status_subtext}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Render 3 Key Metric Cards: Probability, Risk Band, Decision Threshold
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(
                        f"""
                        <div class="stat-card">
                            <div class="stat-label">Estimated Churn Probability</div>
                            <div class="stat-value">{prob_percent}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(max(0.0, min(1.0, probability)))
                    st.markdown(
                        """
                        <div class="scale-markers">
                            <span>0% (Low)</span>
                            <span>30%</span>
                            <span>60%</span>
                            <span>100% (High)</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        f"""
                        <div class="stat-card">
                            <div class="stat-label">Prediction Risk Level</div>
                            <div class="stat-value" style="color: {risk_color};">{risk_level}</div>
                            <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">
                                Model assessment risk band
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with m3:
                    st.markdown(
                        """
                        <div class="stat-card">
                            <div class="stat-label">Decision Threshold</div>
                            <div class="stat-value" style="color: #334155;">50%</div>
                            <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">
                                Probability &ge; 0.50 classifies as Churn
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Render business explanation panel
                st.markdown(
                    f"""
                    <div class="info-panel">
                        <strong>What this means:</strong> {meaning_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # -------------------------------------------------------------
                # 12. EXPLAINABILITY: TOP CONTRIBUTING FACTORS
                # -------------------------------------------------------------
                # Displays features driving prediction toward churn vs away from churn
                st.markdown('<div class="section-title">Factors Influencing This Prediction</div>', unsafe_allow_html=True)
                fcol1, fcol2 = st.columns(2)

                with fcol1:
                    toward_items = ""
                    if toward_churn:
                        for item in toward_churn:
                            feat = item.get("feature", "")
                            toward_items += f'<div class="factor-item">&bull; {feat}</div>'
                    else:
                        toward_items = '<div class="factor-item" style="color: #94a3b8;">None identified in top factors.</div>'

                    st.markdown(
                        f"""
                        <div class="factor-card" style="border-top: 3px solid #dc2626;">
                            <div class="factor-heading" style="color: #991b1b;">More associated with churn</div>
                            {toward_items}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with fcol2:
                    away_items = ""
                    if away_from_churn:
                        for item in away_from_churn:
                            feat = item.get("feature", "")
                            away_items += f'<div class="factor-item">&bull; {feat}</div>'
                    else:
                        away_items = '<div class="factor-item" style="color: #94a3b8;">None identified in top factors.</div>'

                    st.markdown(
                        f"""
                        <div class="factor-card" style="border-top: 3px solid #166534;">
                            <div class="factor-heading" style="color: #166534;">Less associated with churn</div>
                            {away_items}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Non-causal statistical disclaimer
                st.markdown(
                    '<div class="disclaimer-text">'
                    "Note: These factors represent statistical contributions derived from the trained Logistic Regression model "
                    "for this specific prediction, not causal explanations."
                    "</div>",
                    unsafe_allow_html=True,
                )

                # Technical model metadata footer
                st.markdown(
                    f"""
                    <div class="meta-grid">
                        <div class="meta-cell"><strong>Model:</strong> Logistic Regression</div>
                        <div class="meta-cell"><strong>Model Version:</strong> {model_version}</div>
                        <div class="meta-cell"><strong>Prediction Threshold:</strong> 0.50</div>
                        <div class="meta-cell"><strong>Prediction Source:</strong> FastAPI ML API</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception:
                st.error("Failed to parse prediction response from API.")
        # Error handling for non-200 HTTP responses
        elif response.status_code == 422:
            st.error("Input validation error: The data provided does not match API schema requirements.")
        elif response.status_code == 503:
            st.error("Service unavailable: The machine learning model is currently not loaded.")
        elif response.status_code >= 500:
            st.error("Server error: An internal error occurred on the prediction backend.")
        else:
            st.error(f"Prediction failed with HTTP status code {response.status_code}.")


# Script execution entry point
if __name__ == "__main__":
    main()

