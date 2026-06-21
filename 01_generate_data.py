"""
01_generate_data.py
--------------------
Generates a realistic SYNTHETIC healthcare dataset for the analytics project.
No real patient data is used anywhere — everything here is fabricated using Faker
+ controlled randomness, with intentional real-world messiness (nulls, duplicates,
inconsistent formatting) so the cleaning step in the pipeline has real work to do.

Output: CSV files in /data
  - patients.csv
  - doctors.csv
  - departments.csv
  - admissions.csv
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import timedelta

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

N_PATIENTS = 1200
N_DOCTORS = 45
N_ADMISSIONS = 4000

# ---------------------------------------------------------------------------
# 1. DEPARTMENTS
# ---------------------------------------------------------------------------
departments = [
    "Cardiology", "Neurology", "Orthopedics", "Pediatrics", "Oncology",
    "General Medicine", "Emergency", "Gynecology", "Nephrology", "Pulmonology"
]
dept_df = pd.DataFrame({
    "department_id": range(1, len(departments) + 1),
    "department_name": departments
})

# ---------------------------------------------------------------------------
# 2. DOCTORS
# ---------------------------------------------------------------------------
doctor_rows = []
for i in range(1, N_DOCTORS + 1):
    doctor_rows.append({
        "doctor_id": i,
        "doctor_name": "Dr. " + fake.name(),
        "department_id": random.randint(1, len(departments)),
        "years_experience": random.randint(1, 35),
        "consultation_fee": random.choice([300, 400, 500, 600, 800, 1000, 1500])
    })
doctors_df = pd.DataFrame(doctor_rows)

# ---------------------------------------------------------------------------
# 3. PATIENTS
# ---------------------------------------------------------------------------
genders = ["Male", "Female", "Other"]
blood_groups = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
cities = ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata", "Hyderabad",
          "Pune", "Ahmedabad", "Jaipur", "Lucknow"]

patient_rows = []
for i in range(1, N_PATIENTS + 1):
    age = int(np.clip(np.random.normal(45, 20), 0, 95))
    patient_rows.append({
        "patient_id": i,
        "name": fake.name(),
        "age": age,
        "gender": random.choice(genders),
        "blood_group": random.choice(blood_groups) if random.random() > 0.05 else None,
        "city": random.choice(cities),
        "phone": fake.phone_number() if random.random() > 0.03 else None,
        "registration_date": fake.date_between(start_date="-4y", end_date="-1d")
    })
patients_df = pd.DataFrame(patient_rows)

# inject a few duplicate patients (real-world data quality issue)
dupes = patients_df.sample(15, random_state=1).copy()
patients_df = pd.concat([patients_df, dupes], ignore_index=True)

# ---------------------------------------------------------------------------
# 4. ADMISSIONS (the core fact table)
# ---------------------------------------------------------------------------
diagnoses_by_dept = {
    "Cardiology": ["Hypertension", "Myocardial Infarction", "Arrhythmia", "Heart Failure"],
    "Neurology": ["Migraine", "Epilepsy", "Stroke", "Parkinson's Disease"],
    "Orthopedics": ["Fracture", "Arthritis", "Spinal Injury", "Ligament Tear"],
    "Pediatrics": ["Viral Fever", "Asthma", "Malnutrition", "Bronchitis"],
    "Oncology": ["Breast Cancer", "Lung Cancer", "Leukemia", "Lymphoma"],
    "General Medicine": ["Diabetes", "Typhoid", "Anemia", "Viral Infection"],
    "Emergency": ["Trauma", "Poisoning", "Burns", "Accident Injury"],
    "Gynecology": ["Pregnancy Care", "PCOS", "Menstrual Disorder", "Infertility"],
    "Nephrology": ["Kidney Stones", "Chronic Kidney Disease", "UTI", "Dialysis"],
    "Pulmonology": ["Asthma", "Pneumonia", "Tuberculosis", "COPD"]
}

admission_types = ["Emergency", "Elective", "Referral"]
discharge_status = ["Discharged", "Transferred", "Deceased", "Left Against Advice"]
insurance_types = ["Government", "Private", "None", "Corporate"]

admission_rows = []
for i in range(1, N_ADMISSIONS + 1):
    patient_id = random.randint(1, N_PATIENTS)
    doctor = doctors_df.sample(1).iloc[0]
    dept_name = dept_df.loc[dept_df.department_id == doctor.department_id, "department_name"].values[0]
    diagnosis = random.choice(diagnoses_by_dept[dept_name])

    admit_date = fake.date_between(start_date="-3y", end_date="-1d")
    # length of stay varies by department realism (emergency shorter, oncology longer)
    base_los = {"Emergency": 2, "Oncology": 7, "Cardiology": 5, "Orthopedics": 6}.get(dept_name, 4)
    los = max(1, int(np.random.poisson(base_los)))
    discharge_date = admit_date + timedelta(days=los)

    bill = round(np.random.gamma(shape=2, scale=base_los * 2500), -2)

    admission_rows.append({
        "admission_id": i,
        "patient_id": patient_id,
        "doctor_id": doctor.doctor_id,
        "department_id": doctor.department_id,
        "diagnosis": diagnosis,
        "admission_type": random.choice(admission_types),
        "admission_date": admit_date,
        "discharge_date": discharge_date,
        "length_of_stay": los,
        "billing_amount": bill,
        "insurance_type": random.choice(insurance_types),
        "discharge_status": random.choices(
            discharge_status, weights=[0.85, 0.07, 0.04, 0.04]
        )[0],
        "readmitted_within_30_days": None  # to be computed in cleaning step
    })

admissions_df = pd.DataFrame(admission_rows)

# inject some missing billing amounts and a few negative ages typo-style errors (data quality issues)
admissions_df.loc[admissions_df.sample(frac=0.02, random_state=2).index, "billing_amount"] = None
patients_df.loc[patients_df.sample(5, random_state=3).index, "age"] = -1  # bad data entry

# ---------------------------------------------------------------------------
# 5. INJECT A REAL, LEARNABLE READMISSION RISK SIGNAL
# ---------------------------------------------------------------------------
# Clinically-inspired risk factors (so the later ML model has genuine signal
# to learn instead of pure noise):
#   - Older patients are more likely to be readmitted
#   - Chronic/high-severity diagnoses (heart failure, CKD, COPD, cancers) carry higher risk
#   - Emergency admissions carry higher risk than elective ones
#   - Longer stays correlate with higher risk (more severe initial condition)
#   - "Left Against Advice" discharges carry much higher risk
high_risk_diagnoses = {
    "Heart Failure", "Chronic Kidney Disease", "COPD", "Lung Cancer",
    "Leukemia", "Lymphoma", "Breast Cancer", "Stroke", "Dialysis"
}

age_lookup = patients_df.drop_duplicates(subset="patient_id").set_index("patient_id")["age"]

risk_scores = []
for _, row in admissions_df.iterrows():
    patient_age = age_lookup.get(row["patient_id"], 45)
    if patient_age < 0:
        patient_age = 45

    score = -3.6  # base log-odds -> low baseline probability
    score += 0.022 * patient_age
    score += 1.1 if row["diagnosis"] in high_risk_diagnoses else 0
    score += 0.7 if row["admission_type"] == "Emergency" else 0
    score += 0.08 * row["length_of_stay"]
    score += 1.3 if row["discharge_status"] == "Left Against Advice" else 0
    score += 0.5 if row["discharge_status"] == "Transferred" else 0
    risk_scores.append(score)

risk_scores = np.array(risk_scores)
probabilities = 1 / (1 + np.exp(-risk_scores))  # sigmoid -> probability
admissions_df["_risk_probability"] = probabilities

# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------
dept_df.to_csv("/home/claude/healthcare_analytics/data/departments.csv", index=False)
doctors_df.to_csv("/home/claude/healthcare_analytics/data/doctors.csv", index=False)
patients_df.to_csv("/home/claude/healthcare_analytics/data/patients.csv", index=False)
admissions_df.to_csv("/home/claude/healthcare_analytics/data/admissions.csv", index=False)

print("Data generated:")
print(f"  Departments : {len(dept_df)}")
print(f"  Doctors     : {len(doctors_df)}")
print(f"  Patients    : {len(patients_df)} (incl. 15 intentional duplicates)")
print(f"  Admissions  : {len(admissions_df)}")
