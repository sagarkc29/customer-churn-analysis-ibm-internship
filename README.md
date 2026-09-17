# Customer Churn Analysis and Prediction Using Machine Learning

## Project Overview

This project analyzes customer churn behavior and develops a machine learning model to identify customers with a higher probability of churn.

The project follows the business intelligence flow:

**Data → Cleaning → Analysis → KPIs → Insights → Risk Prediction → Business Action**

## Objective

- Measure customer churn.
- Identify patterns associated with churn.
- Analyze contract, service, payment, tenure and charge characteristics.
- Build a machine learning model for churn prediction.
- Generate customer-level churn risk scores.
- Provide actionable business recommendations.

## Dataset

**Dataset:** Telco Customer Churn

**Source:** Kaggle

**Dataset link:**  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains 7,043 customer records and 21 original columns.

> Note: The final project uses a dataset different from the learning dataset used in the internship masterclasses.

## Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Logistic Regression
- Jupyter Notebook / Python

## Project Workflow

### 1. Data Cleaning
- Convert `TotalCharges` to numeric.
- Handle missing values.
- Remove duplicate rows.
- Standardize text fields.

### 2. Exploratory Data Analysis
The analysis examines:
- Contract type
- Internet service
- Payment method
- Tenure
- Monthly charges
- Churn rate

### 3. Machine Learning
A Logistic Regression classification model is used.

The data is split into training and testing sets using stratification. Numerical variables are scaled and categorical variables are one-hot encoded through a preprocessing pipeline.

### 4. Risk Prediction

Each customer receives:
- Churn probability
- Risk level

Risk levels:
- Low: probability below 40%
- Medium: 40%–70%
- High: above 70%

## Key Findings

The dataset has an overall observed churn rate of **26.54%**.

Important observed patterns include:
- Month-to-month contracts: **42.71% churn**
- One-year contracts: **11.27% churn**
- Two-year contracts: **2.83% churn**
- Fiber optic service: **41.89% churn**
- Electronic check payment: **45.29% churn**
- 0–12 month tenure: **47.44% churn**
- 49–72 month tenure: **9.51% churn**

These are descriptive relationships in the dataset and should not be interpreted as proof of causation.

## Model Performance

The Logistic Regression model produced:

| Metric | Result |
|---|---:|
| Accuracy | 73.81% |
| Precision | 50.43% |
| Recall | 78.34% |
| F1 Score | 61.36% |
| ROC-AUC | 84.16% |

## Business Recommendations

1. Prioritize retention efforts for customers with high predicted churn probability.
2. Focus on early-tenure customers because observed churn is high in the first 12 months.
3. Review month-to-month customers for appropriate retention or contract-conversion strategies.
4. Investigate the electronic-check payment segment.
5. Use risk scores to support targeted customer-retention campaigns.

## How to Run

1. Install Python 3.10+.
2. Place these files in the same folder:
   - `customer_churn_analysis.py`
   - `WA_Fn-UseC_-Telco-Customer-Churn.csv`
   - `requirements.txt`
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run:

```bash
python customer_churn_analysis.py
```

The script creates the cleaned dataset, customer risk predictions and three dashboard images.

## Project Structure

```text
Customer-Churn-Analysis/
│
├── customer_churn_analysis.py
├── requirements.txt
├── README.md
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── telco_customer_churn_cleaned.csv
├── customer_churn_risk_predictions.csv
├── churn_dashboard_page1.png
├── churn_dashboard_page2.png
└── churn_dashboard_page3.png
```
