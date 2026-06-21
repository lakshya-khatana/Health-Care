-- ============================================================================
-- healthcare_analysis_queries.sql
-- Core business questions answered using SQL on the healthcare.db SQLite DB
-- Run with: sqlite3 data/healthcare.db < sql/analysis_queries.sql
-- ============================================================================

-- 1. Total patients, admissions, and revenue overview
SELECT
    COUNT(DISTINCT patient_id) AS total_patients,
    COUNT(*) AS total_admissions,
    ROUND(SUM(billing_amount), 2) AS total_revenue,
    ROUND(AVG(billing_amount), 2) AS avg_bill_per_admission,
    ROUND(AVG(length_of_stay), 2) AS avg_length_of_stay
FROM admissions;

-- 2. Department-wise performance: admissions, revenue, avg stay, readmission rate
SELECT
    d.department_name,
    COUNT(a.admission_id) AS total_admissions,
    ROUND(SUM(a.billing_amount), 2) AS total_revenue,
    ROUND(AVG(a.length_of_stay), 2) AS avg_length_of_stay,
    ROUND(AVG(a.readmitted_within_30_days) * 100, 2) AS readmission_rate_pct
FROM admissions a
JOIN departments d ON a.department_id = d.department_id
GROUP BY d.department_name
ORDER BY total_revenue DESC;

-- 3. Top 10 doctors by number of patients treated
SELECT
    doc.doctor_name,
    dep.department_name,
    COUNT(a.admission_id) AS patients_treated,
    ROUND(AVG(a.billing_amount), 2) AS avg_bill_generated
FROM admissions a
JOIN doctors doc ON a.doctor_id = doc.doctor_id
JOIN departments dep ON doc.department_id = dep.department_id
GROUP BY doc.doctor_name, dep.department_name
ORDER BY patients_treated DESC
LIMIT 10;

-- 4. Most common diagnoses overall
SELECT
    diagnosis,
    COUNT(*) AS occurrence_count,
    ROUND(AVG(billing_amount), 2) AS avg_bill,
    ROUND(AVG(length_of_stay), 2) AS avg_stay_days
FROM admissions
GROUP BY diagnosis
ORDER BY occurrence_count DESC
LIMIT 10;

-- 5. Monthly admission trend (seasonality check)
SELECT
    strftime('%Y-%m', admission_date) AS month,
    COUNT(*) AS admissions_count,
    ROUND(SUM(billing_amount), 2) AS revenue
FROM admissions
GROUP BY month
ORDER BY month;

-- 6. Age-group wise disease burden
SELECT
    p.age_group,
    COUNT(a.admission_id) AS total_admissions,
    ROUND(AVG(a.billing_amount), 2) AS avg_bill
FROM admissions a
JOIN patients p ON a.patient_id = p.patient_id
GROUP BY p.age_group
ORDER BY total_admissions DESC;

-- 7. Insurance type vs average billing (who pays more out of pocket?)
SELECT
    insurance_type,
    COUNT(*) AS admissions_count,
    ROUND(AVG(billing_amount), 2) AS avg_bill,
    ROUND(SUM(billing_amount), 2) AS total_revenue
FROM admissions
GROUP BY insurance_type
ORDER BY total_revenue DESC;

-- 8. Patients with the highest number of readmissions (high-risk patients)
SELECT
    p.patient_id,
    p.name,
    p.age,
    COUNT(a.admission_id) AS total_admissions,
    SUM(a.readmitted_within_30_days) AS readmission_count
FROM admissions a
JOIN patients p ON a.patient_id = p.patient_id
GROUP BY p.patient_id, p.name, p.age
HAVING total_admissions > 1
ORDER BY readmission_count DESC, total_admissions DESC
LIMIT 15;

-- 9. Discharge status breakdown by department (quality/outcome indicator)
SELECT
    d.department_name,
    a.discharge_status,
    COUNT(*) AS count_cases
FROM admissions a
JOIN departments d ON a.department_id = d.department_id
GROUP BY d.department_name, a.discharge_status
ORDER BY d.department_name, count_cases DESC;

-- 10. Doctor workload vs experience (are senior doctors handling more patients?)
SELECT
    doc.doctor_name,
    doc.years_experience,
    COUNT(a.admission_id) AS patients_handled
FROM doctors doc
LEFT JOIN admissions a ON doc.doctor_id = a.doctor_id
GROUP BY doc.doctor_name, doc.years_experience
ORDER BY doc.years_experience DESC;
