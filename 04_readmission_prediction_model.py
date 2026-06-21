"""
04_readmission_prediction_model.py
------------------------------------
Builds a classification model to predict 30-day readmission risk.
This is the ML add-on that gives the project an extra edge — shows
feature engineering, model comparison, and evaluation, not just one model.

Output:
  - outputs/model_comparison.png
  - outputs/feature_importance.png
  - outputs/confusion_matrix.png
  - data/readmission_model.pkl (saved best model, used by Streamlit dashboard)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

DATA_DIR = "/home/claude/healthcare_analytics/data"
OUT_DIR = "/home/claude/healthcare_analytics/outputs"

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
conn = sqlite3.connect(f"{DATA_DIR}/healthcare.db")
admissions = pd.read_sql("SELECT * FROM admissions", conn, parse_dates=["admission_date", "discharge_date"])
patients = pd.read_sql("SELECT * FROM patients", conn)
conn.close()

df = admissions.merge(patients[["patient_id", "age", "gender", "blood_group"]], on="patient_id")

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------
df["admission_month"] = df["admission_date"].dt.month
df["is_emergency"] = (df["admission_type"] == "Emergency").astype(int)

features = [
    "age", "length_of_stay", "billing_amount", "department_id",
    "admission_month", "is_emergency", "gender", "insurance_type", "discharge_status"
]
target = "readmitted_within_30_days"

model_df = df[features + [target]].copy()

# Encode categoricals
encoders = {}
for col in ["gender", "insurance_type", "discharge_status"]:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))
    encoders[col] = le

X = model_df[features]
y = model_df[target]

print("Class balance:\n", y.value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features for Logistic Regression (tree models don't need this, but it
# doesn't hurt them and keeps the pipeline simple to apply consistently)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# ---------------------------------------------------------------------------
# TRAIN & COMPARE MODELS
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42),
}

results = []
trained_models = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1-Score": f1_score(y_test, preds, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba)
    })
    trained_models[name] = model

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("\nModel Comparison:\n", results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print(f"\nBest model: {best_model_name}")

# ---------------------------------------------------------------------------
# SAVE MODEL COMPARISON CHART
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
results_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(data=results_melted, x="Metric", y="Score", hue="Model")
plt.title("Model Comparison: Readmission Prediction")
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_model_comparison.png")
plt.close()

# ---------------------------------------------------------------------------
# FEATURE IMPORTANCE (Random Forest)
# ---------------------------------------------------------------------------
rf_model = trained_models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette="mako", legend=False)
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/08_feature_importance.png")
plt.close()

# ---------------------------------------------------------------------------
# CONFUSION MATRIX (best model)
# ---------------------------------------------------------------------------
preds_best = best_model.predict(X_test_scaled if best_model_name == "Logistic Regression" else X_test)
cm = confusion_matrix(y_test, preds_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Readmitted", "Readmitted"],
            yticklabels=["Not Readmitted", "Readmitted"])
plt.title(f"Confusion Matrix — {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/09_confusion_matrix.png")
plt.close()

print("\nClassification Report (best model):")
print(classification_report(y_test, preds_best, target_names=["Not Readmitted", "Readmitted"]))

# ---------------------------------------------------------------------------
# SAVE MODEL + ENCODERS FOR DASHBOARD USE
# ---------------------------------------------------------------------------
joblib.dump({
    "model": best_model,
    "encoders": encoders,
    "scaler": scaler,
    "features": features,
    "model_name": best_model_name
}, f"{DATA_DIR}/readmission_model.pkl")

results_df.to_csv(f"{OUT_DIR}/model_comparison_results.csv", index=False)

print(f"\nModel saved to {DATA_DIR}/readmission_model.pkl")
print("Charts saved: 07_model_comparison.png, 08_feature_importance.png, 09_confusion_matrix.png")
