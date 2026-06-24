# Credit Card Fraud Detection System

## Overview

This project is an end-to-end machine learning system for detecting fraudulent credit card transactions using XGBoost. It includes full data preprocessing, feature engineering, model training, threshold optimization, explainability using SHAP, and deployment using FastAPI and Docker.

The goal is to build a reliable system that can detect fraudulent transactions in real-time while handling severe class imbalance.

## Problem Statement

Credit card fraud is a highly imbalanced classification problem where fraudulent transactions are extremely rare compared to legitimate ones. The challenge is to maximize fraud detection (recall) while minimizing false alerts (precision trade-off).

## Dataset
Full dataset is not included due to size constraints.
A 1000-row stratified sample is provided for demonstration purposes.

The dataset includes transaction-level information such as:

- Transaction amount
- Merchant category
- Customer demographics
- Geographic location
- Time-based features

Target variable:
- is_fraud (0 = Legitimate, 1 = Fraud)

## Feature Engineering

The following features were created:

- age from date of birth
- hour, day, month, dayofweek from transaction timestamp
- Removal of unnecessary identifiers (names, IDs, addresses)

## Model Pipeline

The system uses a full sklearn Pipeline:

- OneHotEncoding for categorical variables
- Numerical preprocessing (imputation/scaling)
- ColumnTransformer for unified preprocessing
- XGBoost classifier for prediction
- Class imbalance handling using scale_pos_weight

## Model Configuration

- n_estimators = 600
- max_depth = 5
- learning_rate = 0.02
- subsample = 0.9
- colsample_bytree = 0.9
- random_state = 42

## Threshold Optimization

Instead of using the default threshold (0.5), the optimal threshold was selected using Precision-Recall curve and F1-score optimization.

Best Threshold:
0.9578745

## Cross Validation

5-Fold Stratified Cross Validation was used to evaluate model stability.

Results:

- Mean ROC-AUC: 0.9963
- Standard Deviation: 0.0006

The model demonstrated highly consistent performance across different data splits, indicating strong generalization ability.

## Model Performance

AUC Score: 0.9963

Fraud Class:
- Precision: 0.83
- Recall: 0.80
- F1-score: 0.82

Confusion Matrix:

<img width="466" height="372" alt="image" src="https://github.com/user-attachments/assets/f5a5e617-2359-44d7-8f9d-cbc1cc31776a" />

SHAP:

<img width="685" height="786" alt="image" src="https://github.com/user-attachments/assets/8fffc8e2-5ce8-44a4-8613-fd4e65bb0014" />




## Business Insights

Transaction amount (amt) is the most important feature for fraud detection, indicating that high-value transactions are more likely to be fraudulent.

Merchant categories such as gas_transport, grocery_net, food_dining, and shopping_net are highly indicative of fraud behavior.

Transaction time (hour) plays an important role in fraud patterns.

Certain cities show higher fraud tendency, indicating geographic patterns.

Online transactions are more likely to be fraudulent than offline transactions.

## Recommendations

Apply strict monitoring on high-value transactions.

Add additional verification steps such as OTP or multi-factor authentication.

Increase monitoring for high-risk merchant categories.

Introduce time-based fraud detection rules for unusual hours.

Adopt a risk scoring system instead of binary classification.

## Explainability

Model interpretability is achieved using:

- Feature importance from XGBoost
- SHAP values for feature-level impact analysis

## Deployment

The model is deployed using FastAPI for real-time predictions.

## Docker

The project is containerized using Docker.

Build image:
docker build -t fraud-detection-api .

Run container:
docker run -p 8000:8000 fraud-detection-api

API URL:
http://localhost:8000

## Saved Artifacts

- model.pkl → Full ML pipeline
- features.pkl → Feature order reference
- threshold.pkl → Optimal decision threshold

## How to Run

pip install -r requirements.txt

uvicorn app:app --reload

## Example Request

{
  "merchant": "amazon",
  "category": "shopping_net",
  "gender": "M",
  "city": "Cairo",
  "state": "CA",
  "job": "engineer",
  "amt": 500,
  "zip": 12345,
  "lat": 30.0,
  "long": 31.0,
  "city_pop": 100000,
  "unix_time": 1234567890,
  "merch_lat": 30.1,
  "merch_long": 31.1,
  "age": 35,
  "hour": 23,
  "day": 12,
  "month": 6,
  "dayofweek": 5
}

## Output Example

{
  "fraud_probability": 0.5304436683654785,
  "fraud_prediction": 0,
  "threshold_used": 0.9578744769096375
}

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, FastAPI, Docker, Joblib

## Conclusion

This project demonstrates a complete machine learning lifecycle from data preprocessing to deployment. It effectively detects fraudulent transactions using optimized thresholding and provides interpretability for real-world decision-making.

The system is production-ready and can be extended with monitoring, logging, and real-time streaming integrations.
