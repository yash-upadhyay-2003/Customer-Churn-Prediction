# Customer Churn Prediction

An end-to-end machine learning application that predicts whether a telecom customer is likely to churn based on customer, account, and service information.

## Project Overview

The project follows the complete machine learning workflow:

Raw Data
→ Data Cleaning
→ EDA
→ Correlation Analysis
→ Feature Engineering
→ Feature Selection
→ Train/Test Split
→ Model Training
→ Model Evaluation
→ Hyperparameter Tuning
→ Final Model Selection
→ Model Serialization
→ FastAPI
→ Streamlit

## Final Model

The final selected model is:

- Model: Logistic Regression
- Scikit-learn: 1.6.1
- Model Version: 1.0.0
- Evaluation metrics:
  - Accuracy: 80.41%
  - Precision: 66.01%
  - Recall: 54.16%
  - F1 Score: 59.50%
  - ROC-AUC: 84.08%

The model was selected using cross-validation performance, F1, ROC-AUC, overfitting behaviour, and model complexity rather than selecting a model based only on accuracy.

## Dataset

The dataset contains telecom customer information including:

- Customer demographics
- Tenure
- Phone and internet services
- Security and support services
- Contract information
- Billing information
- Monthly and total charges

Target variable:

`Churn`

Customer identifiers are excluded from model input.

## Feature Engineering

The saved prediction pipeline creates the following engineered features:

- TenureYears
- AvgMonthlySpend
- ActiveServiceCount
- IsMonthToMonth
- IsAutoPayment

The complete feature engineering and preprocessing pipeline is stored together with the trained model.

## Architecture

```text
Streamlit UI
     |
     | HTTP POST /predict
     v
FastAPI Backend
     |
     v
Saved ML Pipeline
     |
     ├── Feature Engineering
     ├── Missing Value Handling
     ├── Scaling
     ├── Encoding
     └── Logistic Regression
     |
     v
Prediction + Probability + Explanation