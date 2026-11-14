
import os
import numpy as np
import pandas as pd
import joblib

from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, '..', 'Models',
    'meridian_gradient_boosting_model.joblib'
)

loaded_model = joblib.load(MODEL_PATH)

# Define the model's expected input features
numerical_features = [
    'historical_frequency',
    'historical_monetary',
    'historical_avg_order_value',
    'historical_recency',
    'historical_quantity',
    'customer_tenure_days',
    'historical_ticket_count',
    'historical_avg_resolution_hours',
    'historical_avg_satisfaction',
    'historical_unique_issue_categories',
    'historical_missing_satisfaction_count'
]

categorical_features = [
    'membership_tier',
    'preferred_channel',
    'state',
    'marketing_opt_in',
    'referral_source'
]

required_features = numerical_features + categorical_features

allowed_categories = {
    'membership_tier': ['Bronze', 'Gold', 'Platinum', 'Silver'],
    'preferred_channel': ['In-Store', 'Mobile App', 'Online'],
    'state': [
        'Arizona', 'California', 'Florida', 'Georgia', 'Illinois',
        'Indiana', 'Maryland', 'Massachusetts', 'Michigan', 'Missouri',
        'New Jersey', 'New York', 'North Carolina', 'Ohio',
        'Pennsylvania', 'Tennessee', 'Texas', 'Virginia',
        'Washington', 'Wisconsin'
    ],
    'marketing_opt_in': ['No', 'Unknown', 'Yes'],
    'referral_source': [
        'Email Campaign', 'Friend Referral', 'Influencer',
        'Search Engine', 'Social Media', 'Tv Ad',
        'Unknown / Not Recorded', 'Walk-In'
    ]
}

PREDICTION_THRESHOLD = 0.60


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Meridian Retail Group Customer Churn Prediction API',
        'status': 'API is running',
        'prediction_endpoint': '/predict'
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        customer_data = request.get_json(silent=True)

        if not isinstance(customer_data, dict):
            return jsonify({
                'error': 'Request must contain a valid JSON object.'
            }), 400

        missing_features = [
            feature for feature in required_features
            if feature not in customer_data
        ]

        if missing_features:
            return jsonify({
                'error': 'Some required customer features are missing.',
                'missing_features': missing_features
            }), 400

        validated_data = customer_data.copy()
        invalid_numerical = []

        for feature in numerical_features:
            value = customer_data[feature]

            if value is None:
                continue

            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                invalid_numerical.append(feature)
                continue

            if not np.isfinite(value) or value < 0:
                invalid_numerical.append(feature)
                continue

            if feature == 'historical_avg_satisfaction':
                if not 1 <= value <= 5:
                    invalid_numerical.append(feature)
                    continue

            validated_data[feature] = float(value)

        if invalid_numerical:
            return jsonify({
                'error': 'Some numerical values are invalid.',
                'invalid_features': invalid_numerical
            }), 400

        invalid_categories = {}

        for feature, allowed_values in allowed_categories.items():
            value = customer_data[feature]

            if value is None:
                continue

            if not isinstance(value, str) or value not in allowed_values:
                invalid_categories[feature] = {
                    'received_value': value,
                    'allowed_values': allowed_values
                }

        if invalid_categories:
            return jsonify({
                'error': 'Some categorical values are not recognised.',
                'invalid_categories': invalid_categories
            }), 400

        customer_df = pd.DataFrame(
            [validated_data],
            columns=required_features
        )

        churn_probability = loaded_model.predict_proba(
            customer_df
        )[0][1]

        churn_prediction = int(
            churn_probability >= PREDICTION_THRESHOLD
        )

        return jsonify({
            'predicted_churn': churn_prediction,
            'churn_probability': round(float(churn_probability), 4),
            'risk_level': 'High' if churn_prediction == 1 else 'Low',
            'prediction_threshold': PREDICTION_THRESHOLD
        }), 200

    except Exception:
        app.logger.exception('Prediction request failed.')
        return jsonify({
            'error': 'An unexpected error occurred while processing the prediction.'
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
