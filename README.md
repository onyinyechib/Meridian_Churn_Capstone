# Reducing Customer Churn at Meridian Retail Group
## An End-to-End Data Science Investigation

## 1. Project Overview
This project investigates customer churn at Meridian Retail Group using customer, product, order, order-item and customer-support data. It applies data cleaning, SQL, exploratory analysis, visualisation, feature engineering and machine learning to develop a customer churn prediction solution.

## 2. Business Problem
Customer churn occurs when customers stop purchasing from a business. High churn can reduce revenue, weaken customer relationships and increase customer acquisition costs. Meridian needs a data-driven approach to identify customers at risk and support retention decisions.

## 3. Project Objectives
- Understand the business problem and define customer churn.
- Acquire and assess the five datasets.
- Clean and integrate the data.
- Explore purchasing and customer-support patterns.
- Engineer historical customer features.
- Train and evaluate classification models.
- Develop a churn prediction API.
- Translate findings into business recommendations.

## 4. Datasets
| Dataset | Description |
|---|---|
| customers.csv | Customer profiles |
| products.csv | Product catalogue |
| orders.csv | Order-level transactions |
| order_items.csv | Products and quantities within orders |
| support_tickets.csv | Customer-support interactions |

Raw datasets are stored in `Data/Raw/`; cleaned datasets are stored in `Data/Cleaned/`.

## 5. Project Structure
```text
Meridian_Churn_Capstone/
├── API/
├── Data/Raw/
├── Data/Cleaned/
├── Excel/
├── Models/
├── Notebooks/
├── Outputs/
├── Reports/
├── Sql/
├── Visualizations/
├── Src/
└── README.md
```

## 6. Churn Definition and Modelling Population
Customers with no completed orders during 2 January to 30 June 2026 were classified as churned. The prediction cut-off was 2 January 2026, and historical features were constructed using information available before that date.

The eligible modelling population contained 5,564 customers: 3,996 churned (71.82%) and 1,568 active (28.18%).

## 7. Machine-Learning Models
The investigation compared a Dummy Classifier, Logistic Regression, Decision Tree, Random Forest and Gradient Boosting. A tuned Gradient Boosting model was selected for the standalone API.

Numerical features were imputed using the median and scaled. Categorical features were imputed using the most frequent category and one-hot encoded.

## 8. Model Evaluation
The tuned Gradient Boosting model achieved a test ROC-AUC of approximately 0.7942 at the default 0.50 threshold.

At the 0.60 threshold used by the API, the exploratory test results were:
- Accuracy: 78.08%.
- Churn precision: 82.84%.
- Churn recall: 87.61%.
- Churn F1-score: 85.16%.
- Active-customer recall: 53.82%.

The test set was also examined during threshold selection. Therefore, these results should not be described as a completely untouched final evaluation.

## 9. Important Findings
Historical recency accounted for approximately 48.34% of the tuned model's feature importance. Customer tenure and historical purchase frequency were also influential. Together, these three features accounted for approximately 84.52% of feature importance.

Feature importance indicates predictive contribution within the model; it does not establish causation.

## 10. Business Recommendations
- Monitor customer purchase recency.
- Develop engagement strategies for customers with declining purchasing activity.
- Review retention strategies for customers with low purchase frequency.
- Consider customer tenure when designing onboarding and loyalty programmes.
- Monitor customer-support experience and satisfaction.
- Evaluate retention campaigns through measured outcomes.

## 11. Prediction API
The `API/` directory contains a standalone Flask application.

- `GET /` provides API status information.
- `POST /predict` accepts customer features in JSON and returns a churn prediction, probability, risk level and threshold.

The API validates missing features, numerical values and categorical values. See `API/API_README.md` for detailed instructions.

The API has been tested using actual HTTP requests. Test results are saved in `API/api_http_test_evidence.json`.

## 12. Limitations
- The datasets represent a fictional retail business.
- Churn classification depends on the selected observation period.
- The eligible population has an imbalanced churn distribution.
- Predictions are probabilistic, not certain outcomes.
- Feature importance does not establish causal relationships.
- Threshold selection involves trade-offs between identifying churn and recognising active customers.
- An independent future evaluation would provide stronger evidence of generalisation.

## 13. Reproducibility
The `Notebooks/` directory contains the saved analysis notebooks. The `Models/` directory contains the trained model. The `API/` directory contains the application, documentation, sample request and HTTP test evidence.

**Primary model:** Tuned Gradient Boosting

**API framework:** Flask

**Project type:** End-to-end data science investigation

## 14. Installation and Running the API

### Requirements
- Python 3.10 or later is recommended.
- Preserve the project folder structure, particularly the relationship between API/ and Models/.

### Install dependencies
Open a terminal in the project root directory and run:

    pip install -r requirements.txt

### Start the Flask API
Open a terminal in the project root directory and run:

    cd API
    python app.py

The application listens on port 5000.

### Test the API
Open the home endpoint in a browser:

    http://127.0.0.1:5000/

To make a prediction, send a POST request to:

    http://127.0.0.1:5000/predict

Use API/sample_request.json as the request body and set the Content-Type header to application/json.
See API/API_README.md for the complete input specification, sample response and error-handling details.

### Model threshold
The API currently uses a classification threshold of 0.60 rather than 0.77. Although 0.77 produced the highest macro recall during training-only cross-validation, the 0.60 threshold was retained for the API because Meridian's immediate business priority is to identify as many potentially churning customers as possible. On the test set, the 0.60 threshold identified 700 of 799 churned customers (87.61%), compared with 530 (66.33%) at 0.77. However, the 0.60 threshold also flagged more active customers: 145 compared with 69 at 0.77. This is a business trade-off that Meridian should review against its retention capacity and the relative costs of false positives and false negatives.

### Evaluation limitation
The test dataset was examined during earlier threshold comparisons. Although the threshold of 0.77 was selected using training-only cross-validation, the test results are not a completely untouched independent evaluation. A future evaluation using a new, unseen time period is recommended.
