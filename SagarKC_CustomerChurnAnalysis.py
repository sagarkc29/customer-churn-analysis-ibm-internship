"""
Customer Churn Analysis and Prediction Using Machine Learning
IBM SkillsBuild Academic Internship – Data Analytics with AI

Input:
    WA_Fn-UseC_-Telco-Customer-Churn.csv

Outputs:
    telco_customer_churn_cleaned.csv
    customer_churn_risk_predictions.csv
    churn_dashboard_page1.png
    churn_dashboard_page2.png
    churn_dashboard_page3.png
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

INPUT_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv(INPUT_FILE)
df.columns = [c.strip() for c in df.columns]

print("=" * 60)
print("CUSTOMER CHURN ANALYSIS AND PREDICTION")
print("=" * 60)
print(f"Original shape: {df.shape}")

# ============================================================
# 2. DATA CLEANING
# ============================================================
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# The blank TotalCharges records correspond to zero-tenure customers.
df.loc[
    df["TotalCharges"].isna() & df["tenure"].eq(0),
    "TotalCharges"
] = 0

# Safety fallback if any missing values remain.
if df["TotalCharges"].isna().any():
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

df = df.drop_duplicates().reset_index(drop=True)

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

df.to_csv("telco_customer_churn_cleaned.csv", index=False)

print(f"Cleaned shape: {df.shape}")
print(f"Missing values: {df.isna().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# ============================================================
# 3. KPI CALCULATIONS
# ============================================================
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

total_customers = len(df)
churned_customers = int(df["ChurnFlag"].sum())
churn_rate = df["ChurnFlag"].mean() * 100
avg_monthly_charges = df["MonthlyCharges"].mean()
avg_tenure = df["tenure"].mean()

print("\nKEY PERFORMANCE INDICATORS")
print(f"Total Customers: {total_customers:,}")
print(f"Churned Customers: {churned_customers:,}")
print(f"Churn Rate: {churn_rate:.2f}%")
print(f"Average Monthly Charges: {avg_monthly_charges:.2f}")
print(f"Average Tenure: {avg_tenure:.2f} months")

# ============================================================
# 4. EDA / DERIVED FIELDS
# ============================================================
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0–12 months", "13–24 months", "25–48 months", "49–72 months"]
)

contract_churn = (
    df.groupby("Contract")["ChurnFlag"].mean().mul(100).sort_values(ascending=False)
)

internet_churn = (
    df.groupby("InternetService")["ChurnFlag"].mean().mul(100).sort_values(ascending=False)
)

payment_churn = (
    df.groupby("PaymentMethod")["ChurnFlag"].mean().mul(100).sort_values(ascending=False)
)

tenure_churn = (
    df.groupby("TenureGroup", observed=True)["ChurnFlag"]
    .mean().mul(100)
)

# ============================================================
# 5. PAGE 1 — EXECUTIVE OVERVIEW
# ============================================================
fig = plt.figure(figsize=(12, 8))
fig.suptitle("Customer Churn Analysis — Executive Overview",
             fontsize=18, fontweight="bold")

kpi_data = [
    ("Total Customers", f"{total_customers:,}"),
    ("Churned Customers", f"{churned_customers:,}"),
    ("Churn Rate", f"{churn_rate:.2f}%"),
    ("Avg Monthly Charges", f"{avg_monthly_charges:.2f}"),
    ("Avg Tenure", f"{avg_tenure:.2f} months"),
]

for i, (label, value) in enumerate(kpi_data):
    ax = fig.add_axes([0.05 + i * 0.19, 0.72, 0.16, 0.15])
    ax.axis("off")
    ax.text(0.5, 0.60, value, ha="center", va="center",
            fontsize=17, fontweight="bold")
    ax.text(0.5, 0.15, label, ha="center", va="center", fontsize=9)

ax1 = fig.add_axes([0.08, 0.12, 0.38, 0.43])
df["Churn"].value_counts().reindex(["No", "Yes"]).plot(
    kind="bar", ax=ax1
)
ax1.set_title("Customer Churn Distribution")
ax1.set_xlabel("Churn")
ax1.set_ylabel("Customers")
ax1.tick_params(axis="x", rotation=0)

ax2 = fig.add_axes([0.56, 0.12, 0.36, 0.43])
contract_churn.sort_values().plot(kind="barh", ax=ax2)
ax2.set_title("Churn Rate by Contract")
ax2.set_xlabel("Churn Rate (%)")
ax2.set_ylabel("")

fig.savefig("churn_dashboard_page1.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 6. PAGE 2 — CUSTOMER & SERVICE ANALYSIS
# ============================================================
fig = plt.figure(figsize=(12, 8))
fig.suptitle("Customer Churn Analysis — Customer & Service Analysis",
             fontsize=18, fontweight="bold")

ax1 = fig.add_axes([0.08, 0.56, 0.38, 0.30])
internet_churn.sort_values().plot(kind="barh", ax=ax1)
ax1.set_title("Churn Rate by Internet Service")
ax1.set_xlabel("Churn Rate (%)")

ax2 = fig.add_axes([0.56, 0.56, 0.38, 0.30])
payment_churn.sort_values().plot(kind="barh", ax=ax2)
ax2.set_title("Churn Rate by Payment Method")
ax2.set_xlabel("Churn Rate (%)")

ax3 = fig.add_axes([0.08, 0.12, 0.38, 0.30])
tenure_churn.plot(kind="bar", ax=ax3)
ax3.set_title("Churn Rate by Tenure Group")
ax3.set_xlabel("Tenure")
ax3.set_ylabel("Churn Rate (%)")
ax3.tick_params(axis="x", rotation=25)

ax4 = fig.add_axes([0.56, 0.12, 0.38, 0.30])
df.groupby("Churn")["MonthlyCharges"].mean().reindex(["No", "Yes"]).plot(
    kind="bar", ax=ax4
)
ax4.set_title("Average Monthly Charges by Churn")
ax4.set_xlabel("Churn")
ax4.set_ylabel("Average Monthly Charges")
ax4.tick_params(axis="x", rotation=0)

fig.savefig("churn_dashboard_page2.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 7. MACHINE LEARNING
# ============================================================
y = df["ChurnFlag"]
X = df.drop(columns=["customerID", "Churn", "ChurnFlag",
                     "TenureGroup"], errors="ignore")

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
])

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nMODEL PERFORMANCE")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

# ============================================================
# 8. CUSTOMER RISK PREDICTION
# ============================================================
risk = df[["customerID"]].copy()
risk["ChurnProbability"] = pipeline.predict_proba(X)[:, 1]
risk["RiskLevel"] = pd.cut(
    risk["ChurnProbability"],
    bins=[-0.001, 0.40, 0.70, 1.0],
    labels=["Low", "Medium", "High"]
)
risk = risk.sort_values("ChurnProbability", ascending=False)
risk.to_csv("customer_churn_risk_predictions.csv", index=False)

# ============================================================
# 9. PAGE 3 — RISK, PREDICTION & ACTION
# ============================================================
risk_counts = risk["RiskLevel"].value_counts().reindex(
    ["Low", "Medium", "High"]
).fillna(0)

risk_segment = (
    df.groupby(["Contract", "InternetService"])["ChurnFlag"]
    .agg(Customers="count", ChurnRate="mean")
    .reset_index()
)
risk_segment["ChurnRate"] *= 100
risk_segment = risk_segment.sort_values("ChurnRate", ascending=False)

fig = plt.figure(figsize=(12, 8))
fig.suptitle("Customer Churn Analysis — Risk, Prediction & Action",
             fontsize=18, fontweight="bold")

ax1 = fig.add_axes([0.08, 0.55, 0.38, 0.30])
risk_counts.plot(kind="bar", ax=ax1)
ax1.set_title("Predicted Customer Risk Levels")
ax1.set_xlabel("Risk Level")
ax1.set_ylabel("Customers")
ax1.tick_params(axis="x", rotation=0)

ax2 = fig.add_axes([0.56, 0.55, 0.38, 0.30])
risk_segment.head(6).sort_values("ChurnRate").plot(
    x="Contract", y="ChurnRate", kind="barh", ax=ax2, legend=False
)
ax2.set_title("Highest Observed Churn Segments")
ax2.set_xlabel("Observed Churn Rate (%)")
ax2.set_ylabel("Contract / Service Segment")

ax3 = fig.add_axes([0.08, 0.10, 0.86, 0.30])
ax3.axis("off")
actions = (
    "KEY INSIGHTS & RECOMMENDED ACTIONS\n\n"
    "• Prioritize retention efforts for high-probability churn customers.\n"
    "• Investigate early-tenure customers because the 0–12 month group shows high observed churn.\n"
    "• Review month-to-month contracts and identify suitable conversion/retention offers.\n"
    "• Examine electronic-check customers for payment experience or engagement issues.\n"
    "• Use customer risk scores to support targeted, rather than broad, retention campaigns."
)
ax3.text(0.02, 0.95, actions, va="top", fontsize=12, linespacing=1.6)

fig.savefig("churn_dashboard_page3.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print("\nPROJECT COMPLETE")
print("Generated:")
print(" - telco_customer_churn_cleaned.csv")
print(" - customer_churn_risk_predictions.csv")
print(" - churn_dashboard_page1.png")
print(" - churn_dashboard_page2.png")
print(" - churn_dashboard_page3.png")
