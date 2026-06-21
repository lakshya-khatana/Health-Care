"""
02_clean_and_load_db.py
-------------------------
Cleans raw CSVs and loads them into a SQLite database (healthcare.db).
This is the "data engineering" step of the pipeline:
  - Removes duplicate patient records
  - Fixes invalid ages
  - Handles missing billing amounts (median imputation by department)
  - Computes the readmitted_within_30_days flag from admission history
  - Creates a clean star-schema-style SQLite DB for SQL analysis
"""

import pandas as pd
import sqlite3
import numpy as np

DATA_DIR = "/home/claude/healthcare_analytics/data"
DB_PATH = "/home/claude/healthcare_analytics/data/healthcare.db"

# ---------------------------------------------------------------------------
# LOAD RAW
# ---------------------------------------------------------------------------
patients = pd.read_csv(f"{DATA_DIR}/patients.csv")
doctors = pd.read_csv(f"{DATA_DIR}/doctors.csv")
departments = pd.read_csv(f"{DATA_DIR}/departments.csv")
admissions = pd.read_csv(f"{DATA_DIR}/admissions.csv", parse_dates=["admission_date", "discharge_date"])

print("RAW SHAPES:", patients.shape, doctors.shape, departments.shape, admissions.shape)

# ---------------------------------------------------------------------------
# CLEAN PATIENTS
# ---------------------------------------------------------------------------
before = len(patients)
patients = patients.drop_duplicates(subset=["name", "age", "gender", "city"], keep="first")
print(f"Removed {before - len(patients)} duplicate patient rows")

# Fix invalid ages (-1 placeholders) -> impute with median age
invalid_age_mask = patients["age"] < 0
median_age = patients.loc[~invalid_age_mask, "age"].median()
patients.loc[invalid_age_mask, "age"] = median_age
print(f"Fixed {invalid_age_mask.sum()} invalid age entries (imputed with median={median_age})")

# Fill missing blood group / phone with 'Unknown'
patients["blood_group"] = patients["blood_group"].fillna("Unknown")
patients["phone"] = patients["phone"].fillna("Unknown")

# Age bands for easier analysis later
bins = [0, 12, 19, 35, 50, 65, 120]
labels = ["Child", "Teen", "Young Adult", "Adult", "Middle Age", "Senior"]
patients["age_group"] = pd.cut(patients["age"], bins=bins, labels=labels)

# ---------------------------------------------------------------------------
# CLEAN ADMISSIONS
# ---------------------------------------------------------------------------
# Impute missing billing amounts using median billing per department
admissions["billing_amount"] = admissions.groupby("department_id")["billing_amount"] \
    .transform(lambda x: x.fillna(x.median()))

# Recompute length_of_stay defensively from dates (in case of inconsistency)
admissions["length_of_stay"] = (admissions["discharge_date"] - admissions["admission_date"]).dt.days
admissions.loc[admissions["length_of_stay"] < 0, "length_of_stay"] = 1

# ---------------------------------------------------------------------------
# COMPUTE READMISSION FLAG
# Uses the latent risk probability injected during data generation
# (clinically-inspired: age, diagnosis severity, admission type, length of
# stay, discharge status) to decide whether each admission resulted in a
# readmission within 30 days. This keeps the signal genuinely learnable by
# the ML model later, instead of being purely random.
# ---------------------------------------------------------------------------
np.random.seed(7)
if "_risk_probability" in admissions.columns:
    admissions["readmitted_within_30_days"] = (
        np.random.random(len(admissions)) < admissions["_risk_probability"]
    ).astype(int)
    admissions = admissions.drop(columns=["_risk_probability"])
else:
    admissions["readmitted_within_30_days"] = 0

print(f"Readmission rate: {admissions['readmitted_within_30_days'].mean():.2%}")

# ---------------------------------------------------------------------------
# LOAD INTO SQLITE
# ---------------------------------------------------------------------------
conn = sqlite3.connect(DB_PATH)

departments.to_sql("departments", conn, if_exists="replace", index=False)
doctors.to_sql("doctors", conn, if_exists="replace", index=False)
patients.to_sql("patients", conn, if_exists="replace", index=False)
admissions.to_sql("admissions", conn, if_exists="replace", index=False)

# Helpful indexes for query performance
cur = conn.cursor()
cur.execute("CREATE INDEX IF NOT EXISTS idx_adm_patient ON admissions(patient_id)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_adm_doctor ON admissions(doctor_id)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_adm_dept ON admissions(department_id)")
conn.commit()
conn.close()

# also save cleaned CSVs (used by Streamlit + ML scripts, avoids reopening db every time)
patients.to_csv(f"{DATA_DIR}/patients_clean.csv", index=False)
admissions.to_csv(f"{DATA_DIR}/admissions_clean.csv", index=False)

print("\nCLEAN SHAPES:", patients.shape, admissions.shape)
print(f"SQLite DB written to: {DB_PATH}")
