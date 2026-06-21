"""
05_export_for_bi_tools.py
----------------------------
Exports a single denormalized, analysis-ready CSV that can be directly
plugged into Power BI or Tableau as a data source (Get Data > Text/CSV).
"""

import pandas as pd
import sqlite3

DATA_DIR = "/home/claude/healthcare_analytics/data"

conn = sqlite3.connect(f"{DATA_DIR}/healthcare.db")
admissions = pd.read_sql("SELECT * FROM admissions", conn, parse_dates=["admission_date", "discharge_date"])
patients = pd.read_sql("SELECT * FROM patients", conn)
doctors = pd.read_sql("SELECT * FROM doctors", conn)
departments = pd.read_sql("SELECT * FROM departments", conn)
conn.close()

flat = (
    admissions
    .merge(departments, on="department_id")
    .merge(doctors, on=["doctor_id", "department_id"])
    .merge(patients[["patient_id", "name", "age", "age_group", "gender", "city", "blood_group"]],
           on="patient_id", suffixes=("", "_patient"))
)

flat = flat.rename(columns={"name": "patient_name", "doctor_name": "doctor_name"})

flat["admission_month_name"] = flat["admission_date"].dt.month_name()
flat["admission_year"] = flat["admission_date"].dt.year
flat["readmitted_flag"] = flat["readmitted_within_30_days"].map({1: "Yes", 0: "No"})

out_path = f"{DATA_DIR}/healthcare_flat_for_bi.csv"
flat.to_csv(out_path, index=False)

print(f"Flat file exported: {out_path}")
print(f"Shape: {flat.shape}")
print("\nReady to import into Power BI / Tableau:")
print("  Power BI : Get Data -> Text/CSV -> select this file")
print("  Tableau  : Connect -> Text File -> select this file")
