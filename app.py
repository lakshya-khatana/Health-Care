"""
app.py — Healthcare Analytics Dashboard
-----------------------------------------
Interactive Streamlit dashboard for the Healthcare Patient & Hospital
Analytics System. Run with:

    streamlit run dashboard/app.py

Tabs:
  1. Overview        - KPIs and high-level trends
  2. Department View - department-wise deep dive
  3. Patient Insights - demographics and high-risk patients
  4. Readmission Predictor - live ML prediction tool
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import plotly.express as px
import os

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="🏥",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "healthcare.db")
MODEL_PATH = os.path.join(BASE_DIR, "data", "readmission_model.pkl")


@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    admissions = pd.read_sql(
        "SELECT * FROM admissions", conn, parse_dates=["admission_date", "discharge_date"]
    )
    patients = pd.read_sql("SELECT * FROM patients", conn)
    doctors = pd.read_sql("SELECT * FROM doctors", conn)
    departments = pd.read_sql("SELECT * FROM departments", conn)
    conn.close()

    full = (
        admissions
        .merge(departments, on="department_id")
        .merge(doctors[["doctor_id", "doctor_name"]], on="doctor_id")
        .merge(patients[["patient_id", "age", "age_group", "gender", "city"]], on="patient_id")
    )
    return full, patients, doctors, departments


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df, patients_df, doctors_df, departments_df = load_data()
model_bundle = load_model()

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.title("🏥 Filters")
dept_options = ["All"] + sorted(df["department_name"].unique().tolist())
selected_dept = st.sidebar.selectbox("Department", dept_options)

min_date, max_date = df["admission_date"].min(), df["admission_date"].max()
date_range = st.sidebar.date_input(
    "Admission Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

filtered = df.copy()
if selected_dept != "All":
    filtered = filtered[filtered["department_name"] == selected_dept]
if len(date_range) == 2:
    filtered = filtered[
        (filtered["admission_date"] >= pd.Timestamp(date_range[0])) &
        (filtered["admission_date"] <= pd.Timestamp(date_range[1]))
    ]

st.sidebar.markdown("---")
st.sidebar.caption(
    "Built as a final-year Data Analytics project. "
    "Synthetic data — no real patient information is used."
)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("🏥 Healthcare Patient & Hospital Analytics")
st.markdown("End-to-end pipeline: **SQLite → Python (Pandas/SQL) → ML → Streamlit**")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🏢 Department View", "🧑‍🤝‍🧑 Patient Insights", "🤖 Readmission Predictor"]
)

# ---------------------------------------------------------------------------
# TAB 1: OVERVIEW
# ---------------------------------------------------------------------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Admissions", f"{len(filtered):,}")
    col2.metric("Total Revenue", f"₹{filtered['billing_amount'].sum():,.0f}")
    col3.metric("Avg. Length of Stay", f"{filtered['length_of_stay'].mean():.1f} days")
    col4.metric("Readmission Rate", f"{filtered['readmitted_within_30_days'].mean()*100:.1f}%")

    st.markdown("### Monthly Admissions Trend")
    monthly = filtered.copy()
    monthly["month"] = monthly["admission_date"].dt.to_period("M").astype(str)
    monthly_counts = monthly.groupby("month").size().reset_index(name="admissions")
    fig = px.line(monthly_counts, x="month", y="admissions", markers=True)
    fig.update_layout(xaxis_title="Month", yaxis_title="Admissions")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Revenue by Department")
        rev = filtered.groupby("department_name")["billing_amount"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(rev, x="billing_amount", y="department_name", orientation="h", color="billing_amount",
                     color_continuous_scale="Viridis")
        fig.update_layout(yaxis_title="", xaxis_title="Revenue (₹)", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("### Insurance Type Split")
        ins = filtered["insurance_type"].value_counts().reset_index()
        ins.columns = ["insurance_type", "count"]
        fig = px.pie(ins, names="insurance_type", values="count", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2: DEPARTMENT VIEW
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("### Department Performance Summary")
    dept_summary = filtered.groupby("department_name").agg(
        total_admissions=("admission_id", "count"),
        total_revenue=("billing_amount", "sum"),
        avg_stay=("length_of_stay", "mean"),
        readmission_rate=("readmitted_within_30_days", "mean")
    ).reset_index().sort_values("total_revenue", ascending=False)
    dept_summary["readmission_rate"] = (dept_summary["readmission_rate"] * 100).round(1)
    dept_summary["avg_stay"] = dept_summary["avg_stay"].round(1)
    dept_summary["total_revenue"] = dept_summary["total_revenue"].round(0)
    st.dataframe(dept_summary, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Readmission Rate by Department")
        fig = px.bar(dept_summary.sort_values("readmission_rate", ascending=False),
                     x="readmission_rate", y="department_name", orientation="h", color="readmission_rate",
                     color_continuous_scale="Reds")
        fig.update_layout(yaxis_title="", xaxis_title="Readmission Rate (%)", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("### Length of Stay Distribution")
        fig = px.box(filtered, x="department_name", y="length_of_stay", color="department_name")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Length of Stay (days)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top Diagnoses")
    diag = filtered["diagnosis"].value_counts().head(10).reset_index()
    diag.columns = ["diagnosis", "count"]
    fig = px.bar(diag, x="count", y="diagnosis", orientation="h", color="count", color_continuous_scale="Teal")
    fig.update_layout(yaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: PATIENT INSIGHTS
# ---------------------------------------------------------------------------
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Age Distribution")
        fig = px.histogram(filtered, x="age", color="gender", nbins=20, barmode="stack")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("### Admissions by Age Group")
        ag = filtered["age_group"].value_counts().reset_index()
        ag.columns = ["age_group", "count"]
        fig = px.bar(ag, x="age_group", y="count", color="age_group")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Admissions")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### High-Risk Patients (Multiple Readmissions)")
    risk_patients = (
        filtered.groupby(["patient_id"])
        .agg(
            total_admissions=("admission_id", "count"),
            readmissions=("readmitted_within_30_days", "sum"),
            avg_bill=("billing_amount", "mean")
        )
        .reset_index()
    )
    risk_patients = risk_patients[risk_patients["total_admissions"] > 1].sort_values(
        "readmissions", ascending=False
    ).head(15)
    risk_patients = risk_patients.merge(patients_df[["patient_id", "name", "age"]], on="patient_id")
    risk_patients = risk_patients[["name", "age", "total_admissions", "readmissions", "avg_bill"]]
    risk_patients["avg_bill"] = risk_patients["avg_bill"].round(0)
    st.dataframe(risk_patients, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 4: READMISSION PREDICTOR (live ML)
# ---------------------------------------------------------------------------
with tab4:
    st.markdown("### Predict 30-Day Readmission Risk")
    st.caption(f"Model in use: **{model_bundle['model_name']}** (trained on historical admissions)")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Patient Age", 0, 100, 55)
        los = st.slider("Length of Stay (days)", 1, 30, 5)
        billing = st.number_input("Billing Amount (₹)", min_value=0, value=15000, step=500)
    with c2:
        dept_name = st.selectbox("Department", sorted(departments_df["department_name"].unique()))
        admission_month = st.slider("Admission Month", 1, 12, 6)
        is_emergency = st.radio("Admission Type", ["Emergency", "Elective/Referral"]) == "Emergency"
    with c3:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        insurance = st.selectbox("Insurance Type", ["Government", "Private", "Corporate", "None"])
        discharge_status = st.selectbox(
            "Discharge Status", ["Discharged", "Transferred", "Deceased", "Left Against Advice"]
        )

    if st.button("Predict Readmission Risk", type="primary"):
        dept_id = int(departments_df.loc[departments_df["department_name"] == dept_name, "department_id"].values[0])

        encoders = model_bundle["encoders"]
        input_dict = {
            "age": age,
            "length_of_stay": los,
            "billing_amount": billing,
            "department_id": dept_id,
            "admission_month": admission_month,
            "is_emergency": int(is_emergency),
            "gender": encoders["gender"].transform([gender])[0],
            "insurance_type": encoders["insurance_type"].transform([insurance])[0],
            "discharge_status": encoders["discharge_status"].transform([discharge_status])[0],
        }
        input_df = pd.DataFrame([input_dict])[model_bundle["features"]]

        if model_bundle["model_name"] == "Logistic Regression":
            input_df = pd.DataFrame(
                model_bundle["scaler"].transform(input_df), columns=input_df.columns
            )

        proba = model_bundle["model"].predict_proba(input_df)[0, 1]
        pred = int(proba >= 0.5)

        st.markdown("---")
        risk_col, gauge_col = st.columns([1, 2])
        with risk_col:
            if pred == 1:
                st.error(f"⚠️ **High Risk** of readmission within 30 days")
            else:
                st.success(f"✅ **Lower Risk** of readmission within 30 days")
            st.metric("Predicted Probability", f"{proba*100:.1f}%")
        with gauge_col:
            fig = px.bar(
                x=[proba * 100], y=["Risk"], orientation="h", range_x=[0, 100],
                color=[proba * 100], color_continuous_scale=["green", "orange", "red"]
            )
            fig.update_layout(
                showlegend=False, coloraxis_showscale=False, height=150,
                xaxis_title="Readmission Probability (%)", yaxis_title=""
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.caption(
        "⚕️ This tool is built on synthetic data for educational/portfolio purposes only "
        "and is not intended for real clinical decision-making."
    )
