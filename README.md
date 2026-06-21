# 🏥 Healthcare Patient & Hospital Analytics System

An end-to-end data analytics project covering the full pipeline: **data generation/cleaning → SQL database → exploratory analysis → machine learning → interactive dashboard.**

> Built as a final-year data analytics portfolio project. All data is **synthetically generated** (no real patient information) — see [Data & Ethics](#-data--ethics) below.

---

## 📌 Project Overview

This project simulates a hospital's patient and admissions data to answer real operational questions:

- Which departments generate the most revenue, and which have the longest stays?
- What are the most common diagnoses, and how do they vary by age group?
- Which patients are at high risk of being **readmitted within 30 days** — and can we predict it?
- How does insurance type affect billing, and how do admissions trend over time?

It's designed to demonstrate the **full data analyst/analytics engineer skillset**: data engineering, SQL, Python/Pandas analysis, data visualization, machine learning, and dashboarding — not just one isolated skill.

---

## 🧱 Tech Stack

| Layer            | Tools Used                                  |
|-------------------|----------------------------------------------|
| Data Generation   | Python, Faker, NumPy                         |
| Database          | SQLite (easily portable to PostgreSQL/MySQL) |
| Data Cleaning     | Pandas                                       |
| Analysis          | SQL, Pandas                                  |
| Visualization     | Matplotlib, Seaborn, Plotly                  |
| Machine Learning  | Scikit-learn (Logistic Regression, Decision Tree, Random Forest) |
| Dashboard         | Streamlit                                    |
| BI Tool Support   | Flat CSV export for Power BI / Tableau       |

---

## 📂 Project Structure

```
healthcare_analytics/
├── data/                          # Raw, cleaned, and exported data + trained model
│   ├── patients.csv / patients_clean.csv
│   ├── admissions.csv / admissions_clean.csv
│   ├── doctors.csv
│   ├── departments.csv
│   ├── healthcare.db              # SQLite database (single source of truth)
│   ├── healthcare_flat_for_bi.csv # Denormalized export for Power BI/Tableau
│   └── readmission_model.pkl      # Trained ML model + encoders + scaler
├── scripts/
│   ├── 01_generate_data.py        # Synthetic data generation
│   ├── 02_clean_and_load_db.py    # Cleaning + SQLite loading
│   ├── 03_eda_visualizations.py   # EDA charts
│   ├── 04_readmission_prediction_model.py  # ML model training
│   └── 05_export_for_bi_tools.py  # Power BI/Tableau export
├── sql/
│   └── analysis_queries.sql       # 10 business-question SQL queries
├── dashboard/
│   └── app.py                     # Streamlit interactive dashboard
├── outputs/                       # Saved charts (PNG) + model comparison results
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline (in order)
```bash
python scripts/01_generate_data.py            # generates raw CSVs
python scripts/02_clean_and_load_db.py         # cleans + loads into SQLite
python scripts/03_eda_visualizations.py        # generates EDA charts
python scripts/04_readmission_prediction_model.py  # trains ML model
python scripts/05_export_for_bi_tools.py       # exports flat file for BI tools
```

### 3. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

### 4. (Optional) SQL queries
Open `sql/analysis_queries.sql` in any SQLite client (e.g. DB Browser for SQLite) pointed at `data/healthcare.db`, or run via Python:
```python
import sqlite3, pandas as pd
conn = sqlite3.connect("data/healthcare.db")
pd.read_sql("SELECT * FROM admissions LIMIT 5", conn)
```

### 5. (Optional) Power BI / Tableau
Import `data/healthcare_flat_for_bi.csv` directly as a data source.

---

## 🔍 Data Pipeline Details

**1. Generation** — Synthetic patients, doctors, departments, and admissions are generated with realistic distributions (age via normal distribution, length-of-stay via Poisson by department, etc.) and intentional data-quality issues (duplicate patients, missing billing amounts, invalid ages) to simulate real-world messy data.

**2. Cleaning** — Duplicates removed, invalid ages imputed with median, missing billing imputed by department median, age groups derived via binning.

**3. Database** — Loaded into a normalized SQLite schema (`patients`, `doctors`, `departments`, `admissions`) with indexes on foreign keys for query performance.

**4. SQL Analysis** — 10 business-question queries covering revenue, readmission rates, doctor workload, diagnosis frequency, and more (`sql/analysis_queries.sql`).

**5. ML Model** — A 30-day readmission risk classifier. The risk signal is built from clinically-inspired factors (patient age, diagnosis severity, admission type, length of stay, discharge outcome) so the model has genuine, explainable signal to learn — not just noise.

---

## 🤖 Machine Learning: Readmission Prediction

Three models were trained and compared:

| Model               | Accuracy | Precision | Recall | F1   | ROC-AUC |
|----------------------|----------|-----------|--------|------|---------|
| Logistic Regression  | ~0.63    | ~0.30     | ~0.64  | ~0.41| ~0.69   |
| Random Forest        | ~0.74    | ~0.36     | ~0.41  | ~0.38| ~0.68   |
| Decision Tree        | ~0.62    | ~0.25     | ~0.50  | ~0.34| ~0.61   |

*(Exact numbers vary slightly by run/seed — see `outputs/model_comparison_results.csv` for the latest results.)*

**Top predictive features**: patient age, billing amount, length of stay, department, and admission month — consistent with real-world clinical readmission research, where age and stay duration are well-documented risk factors.

The best-performing model (selected by ROC-AUC) is saved and powers the **live prediction tool** in the Streamlit dashboard's "Readmission Predictor" tab.

> **Note on model performance**: An ROC-AUC of ~0.65–0.70 is realistic and expected for this kind of problem — readmission prediction is a genuinely hard task even with real hospital data, where published models typically score in a similar range. This is a deliberate, honest result rather than an inflated one.

---

## 📊 Dashboard Features

The Streamlit dashboard (`dashboard/app.py`) has 4 tabs:

1. **Overview** — KPIs (admissions, revenue, avg. stay, readmission rate), monthly trend, revenue by department, insurance split
2. **Department View** — Department performance table, readmission rates, length-of-stay distribution, top diagnoses
3. **Patient Insights** — Age/gender distribution, high-risk (frequently readmitted) patients
4. **Readmission Predictor** — Live, interactive ML prediction: enter patient details and get a real-time readmission risk score

All views are filterable by **department** and **date range** via the sidebar.

---

## 📈 Key Insights (Sample Findings)

- Oncology and Orthopedics are the highest-revenue departments, while Oncology and Nephrology show the highest 30-day readmission rates — consistent with these being higher-severity, chronic-condition specialties.
- Average length of stay is ~4–5 days, varying significantly by department (Emergency shortest, Oncology longest).
- Older patients and emergency admissions are meaningfully more likely to be readmitted within 30 days.
- ~20% overall readmission rate, in line with real-world hospital readmission benchmarks (typically 10–20%).

---

## 🔒 Data & Ethics

All data in this project is **100% synthetically generated** using the `Faker` library and statistical distributions — no real patient records, hospital data, or PII are used anywhere. This was a deliberate design choice to make the project safely shareable (e.g., on GitHub, in a portfolio) while still producing realistic, analyzable patterns.

---

## 🗣️ How to Talk About This Project (Interview Prep)

A few things worth being ready to explain, since you should understand *why* design choices were made, not just that they exist:

- **Why SQLite over a CSV?** Mirrors a real relational schema (patients/doctors/departments/admissions as separate tables with foreign keys), lets you demonstrate SQL joins/aggregations, and is portable — the same schema would work with PostgreSQL/MySQL with minimal changes.
- **Why inject data quality issues deliberately?** Real-world data is never clean. Showing you can detect and handle duplicates, missing values, and invalid entries is more valuable than working with a pre-cleaned dataset.
- **Why is the ROC-AUC ~0.65-0.70 and not 0.95+?** Because the underlying risk factors (age, diagnosis severity, length of stay, admission type) were deliberately built to *resemble real clinical signal* rather than create an artificially easy/leaky prediction problem. A suspiciously perfect score would actually be a red flag in an interview.
- **What would you improve with more time?** More granular clinical features (lab results, vitals, medication history), time-series modeling of patient history rather than single-admission snapshots, hyperparameter tuning via grid/random search, and deploying the dashboard (e.g., Streamlit Community Cloud) for a live demo link.

---

## 📎 Possible Resume Bullet Points

- *Built an end-to-end healthcare analytics pipeline (SQLite, Python, Scikit-learn, Streamlit) processing 4,000+ patient admission records, including data cleaning, SQL-based analysis, and an interactive multi-tab dashboard.*
- *Designed and trained a 30-day hospital readmission risk classifier (Logistic Regression/Random Forest) achieving ~0.69 ROC-AUC, deployed as a live prediction tool within a Streamlit dashboard.*
- *Wrote 10+ analytical SQL queries to surface department-wise revenue, readmission rates, and doctor performance insights from a relational hospital database.*
