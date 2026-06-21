"""
03_eda_visualizations.py
--------------------------
Exploratory Data Analysis on the cleaned healthcare data.
Generates key charts saved to /outputs as PNGs (used in README / reports / PPT).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA_DIR = "/home/claude/healthcare_analytics/data"
OUT_DIR = "/home/claude/healthcare_analytics/outputs"

conn = sqlite3.connect(f"{DATA_DIR}/healthcare.db")
admissions = pd.read_sql("SELECT * FROM admissions", conn, parse_dates=["admission_date", "discharge_date"])
patients = pd.read_sql("SELECT * FROM patients", conn)
departments = pd.read_sql("SELECT * FROM departments", conn)
doctors = pd.read_sql("SELECT * FROM doctors", conn)
conn.close()

admissions = admissions.merge(departments, on="department_id").merge(
    patients[["patient_id", "age", "age_group", "gender"]], on="patient_id"
)

# ---------------------------------------------------------------------------
# 1. Department-wise revenue
# ---------------------------------------------------------------------------
dept_revenue = admissions.groupby("department_name")["billing_amount"].sum().sort_values(ascending=False)
plt.figure(figsize=(9, 5))
sns.barplot(x=dept_revenue.values, y=dept_revenue.index, hue=dept_revenue.index, palette="viridis", legend=False)
plt.title("Total Revenue by Department")
plt.xlabel("Total Billing Amount (₹)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_revenue_by_department.png")
plt.close()

# ---------------------------------------------------------------------------
# 2. Monthly admission trend
# ---------------------------------------------------------------------------
admissions["month"] = admissions["admission_date"].dt.to_period("M").astype(str)
monthly = admissions.groupby("month").size()
plt.figure(figsize=(11, 5))
monthly.plot(kind="line", marker="o", color="#2a6f97")
plt.title("Monthly Admissions Trend")
plt.xlabel("Month")
plt.ylabel("Number of Admissions")
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_monthly_admissions_trend.png")
plt.close()

# ---------------------------------------------------------------------------
# 3. Age distribution by gender
# ---------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
sns.histplot(data=patients, x="age", hue="gender", multiple="stack", bins=20, palette="Set2")
plt.title("Patient Age Distribution by Gender")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_age_distribution.png")
plt.close()

# ---------------------------------------------------------------------------
# 4. Readmission rate by department
# ---------------------------------------------------------------------------
readmit_dept = admissions.groupby("department_name")["readmitted_within_30_days"].mean().sort_values(ascending=False) * 100
plt.figure(figsize=(9, 5))
sns.barplot(x=readmit_dept.values, y=readmit_dept.index, hue=readmit_dept.index, palette="rocket", legend=False)
plt.title("30-Day Readmission Rate by Department (%)")
plt.xlabel("Readmission Rate (%)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_readmission_rate_by_dept.png")
plt.close()

# ---------------------------------------------------------------------------
# 5. Length of stay distribution
# ---------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
sns.boxplot(data=admissions, x="department_name", y="length_of_stay", hue="department_name", palette="coolwarm", legend=False)
plt.title("Length of Stay by Department")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_length_of_stay_by_dept.png")
plt.close()

# ---------------------------------------------------------------------------
# 6. Insurance type distribution
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 7))
admissions["insurance_type"].value_counts().plot.pie(autopct="%1.1f%%", colors=sns.color_palette("pastel"))
plt.title("Admissions by Insurance Type")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_insurance_distribution.png")
plt.close()

print("Saved 6 charts to:", OUT_DIR)
print("\nKey stats:")
print(f"  Total revenue: ₹{admissions['billing_amount'].sum():,.0f}")
print(f"  Avg length of stay: {admissions['length_of_stay'].mean():.1f} days")
print(f"  Overall readmission rate: {admissions['readmitted_within_30_days'].mean()*100:.1f}%")
print(f"  Top department by revenue: {dept_revenue.index[0]}")
