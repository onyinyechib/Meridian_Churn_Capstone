
# Meridian Retail Group – Customer Churn Prediction API

## 1. Overview

This Flask API uses a trained Gradient Boosting model to estimate
the probability that a Meridian Retail Group customer will churn.

It accepts customer information in JSON format and returns a
churn probability, predicted class, and risk classification.

## 2. Model Information

- Model: Tuned Gradient Boosting Classifier
- Model file: meridian_gradient_boosting_model.joblib
- Prediction threshold: 0.60
- Class 1: Customer classified as likely to churn
- Class 0: Customer not classified as likely to churn

The threshold is provisional and should be reviewed against
business costs before production deployment.

## 3. API Endpoints

### Home endpoint

- Method: GET
- URL: /

Confirms that the API is running.

### Prediction endpoint

- Method: POST
- URL: /predict
- Content-Type: application/json

Accepts customer information and returns a churn prediction.

## 4. Required Input Features

### Numerical features

1. historical_frequency
2. historical_monetary
3. historical_avg_order_value
4. historical_recency
5. historical_quantity
6. customer_tenure_days
7. historical_ticket_count
8. historical_avg_resolution_hours
9. historical_avg_satisfaction
10. historical_unique_issue_categories
11. historical_missing_satisfaction_count

### Categorical features

1. membership_tier
2. preferred_channel
3. state
4. marketing_opt_in
5. referral_source

All 16 features must be included. Use JSON null for unavailable
values where the preprocessing pipeline can impute them.

## 5. Sample Request

See sample_request.json for a complete example.

The request must contain all 16 required features.

## 6. Sample Response

HTTP Status: 200

    {
        "predicted_churn": 0,
        "churn_probability": 0.4382,
        "risk_level": "Low",
        "prediction_threshold": 0.6
    }

The example customer has an estimated churn probability of 43.82%.
This is below the 60% threshold, so the customer is classified
as not likely to churn by the model.

## 7. Error Handling

- HTTP 400: Invalid JSON request.
- HTTP 400: Required features are missing.
- HTTP 400: Invalid numerical values.
- HTTP 400: Unrecognised categorical values.
- HTTP 500: Unexpected server-side prediction error.

Negative numerical values are rejected. Satisfaction scores must
be between 1 and 5. Missing values are allowed where the pipeline
can impute them.

Categorical values are restricted to those represented in the
training dataset.

## 8. Running the API

Project structure:

    Meridian_Churn_Capstone/
        Models/
            meridian_gradient_boosting_model.joblib
        API/
            app.py
            sample_request.json
            API_README.md

Install the required libraries:

    pip install flask pandas numpy scikit-learn joblib

Run the application from the API folder:

    python app.py

The application listens on port 5000.

Home endpoint:

    http://127.0.0.1:5000/

Prediction endpoint:

    http://127.0.0.1:5000/predict

Send a POST request to /predict with the sample JSON request.

## 9. Important Limitations

- The API recognises only categorical values represented in
  the training dataset.
- The 0.60 prediction threshold is provisional.
- Predictions are estimates and should support, not replace,
  customer-retention decisions.
- The application has been tested using Flask's test client.
  It has not yet been deployed as a publicly accessible
  production service.
