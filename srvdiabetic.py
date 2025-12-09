import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier

# =========================
# LOAD MODEL & SCALER
# =========================
scaler = joblib.load("scaler.pkl")
model = CatBoostClassifier()
model.load_model("catboost_diabetes.cbm")

# Medians from TRAINING DATA
SKIN_THICKNESS_MEDIAN = 29
DPF_MEDIAN = 0.47

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# =========================
# HEADER IMAGE + TITLE
# =========================
st.image(
    "https://cdn.pixabay.com/photo/2014/11/12/19/25/diabetes-528678_1280.jpg",
    use_container_width=True
)

st.title("🩺 Diabetes Risk Prediction")
st.write("""
Enter the patient’s medical details below.  
Based on advanced machine-learning, this app estimates the probability of diabetes.  
*(You can leave Skin-fold or Family Risk blank — defaults will be applied automatically.)*
""")

# =========================
# INPUT FORM
# =========================
with st.form(key="patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        Glucose = st.number_input("Glucose (mg/dL)", min_value=50, max_value=300, value=120)
        BloodPressure = st.number_input("Blood Pressure (mmHg)", min_value=30, max_value=200, value=70)
        SkinThickness = st.number_input(
            "Skin Thickness in mm (optional — 0 = unknown)",
            min_value=0, max_value=100, value=0
        )
        Insulin = st.number_input("Insulin (µU/mL)", min_value=10, max_value=300, value=85)

    with col2:
        BMI = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=26.5)
        DiabetesPedigreeFunction = st.number_input(
            "Family Risk Score (optional — 0 = unknown)",
            min_value=0.0, max_value=3.0, value=0.0
        )
        Age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)

    submit = st.form_submit_button("🔍 Predict Diabetes")

# =========================
# PREDICTION BLOCK
# =========================
if submit:
    # ✅ Apply same missing-value logic as training
    if SkinThickness == 0:
        SkinThickness = SKIN_THICKNESS_MEDIAN

    if DiabetesPedigreeFunction == 0:
        DiabetesPedigreeFunction = DPF_MEDIAN

    user_data = pd.DataFrame([{
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age
    }])

    # =========================
    # FEATURE ENGINEERING
    # =========================
    user_data["Age_Group"] = pd.cut(
        user_data["Age"],
        bins=[20, 30, 40, 50, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)

    user_data["BMI_Category"] = pd.cut(
        user_data["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)

    user_data["Glucose_Insulin_Ratio"] = user_data["Glucose"] / user_data["Insulin"]

    # =========================
    # SCALE & PREDICT
    # =========================
    user_data_scaled = scaler.transform(user_data)
    probability = model.predict_proba(user_data_scaled)[0][1]

    # =========================
    # ✅ PROBABILITY GAUGE
    # =========================
   # =========================
# ✅ PURE STREAMLIT RISK PROBABILITY METER (NO HTML)
# =========================
    st.subheader("📊 Risk Probability Meter")

# ✅ Force bar movement based on SEVERITY, not just probability
    if probability < 0.40 and Glucose < 100:
      meter_value = 0.30   # Short bar for low risk
      risk_label = "🟢 Low Risk Zone"

    elif 100 <= Glucose < 126:
     meter_value = 0.50   # Further bar for moderate risk
     risk_label = "🟡 Moderate Risk Zone"

    else:
     meter_value = 0.95   # Almost full bar for high risk
     risk_label = "🔴 High Risk Zone"

    st.progress(meter_value)
    st.caption(f"{risk_label} — Severity Level: {int(meter_value * 100)}%")

    # =========================
    # ✅ RESULT DISPLAY (HYBRID LOGIC)
    # =========================
    st.subheader("✅ Prediction Result")

    if probability < 0.40 and Glucose < 100:
        status = "Non-Diabetic"
        st.success("🟢 Status: Non-Diabetic")

    elif 100 <= Glucose < 126:
        status = "Pre-Diabetic"
        st.warning("🟡 Status: Pre-Diabetic")

    elif probability >= 0.70:
        status = "Diabetic"
        st.error("🔴 Status: Diabetic")

    else:
        status = "Borderline Risk"
        st.warning("🟡 Status: Borderline Risk — Monitor Closely")

    # =========================
    # ✅ LIFESTYLE RECOMMENDATIONS
    # =========================
    st.subheader("🍎 Lifestyle & Health Recommendations")

    if status == "Non-Diabetic":
        st.markdown("""
        ✅ Maintain a healthy routine:
        - 🥗 Balanced diet
        - 🚶 30 minutes walking daily
        - 💧 Stay hydrated
        - 😴 Proper sleep
        """)

    elif status in ["Pre-Diabetic", "Borderline Risk"]:
        st.markdown("""
        ⚠️ Control immediately:
        - 🍭 Reduce sugar & junk food
        - 🏃 45 minutes daily exercise
        - ⚖️ Weight management
        - 🩺 Monthly glucose check
        """)

    else:
        st.markdown("""
        🚨 Immediate action required:
        - 🩺 Consult a doctor
        - 🧪 Daily glucose monitoring
        - 💊 Follow prescribed medication
        - 🥦 Strict diabetic diet
        """)

    # =========================
    # ✅ PATIENT REPORT DOWNLOAD
    # =========================
    report_data = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age,
        "Diabetes_Probability": round(probability, 3),
        "Status": status
    }

    report_df = pd.DataFrame([report_data])

    st.download_button(
        label="⬇️ Download Patient Report (CSV)",
        data=report_df.to_csv(index=False),
        file_name="diabetes_prediction_report.csv",
        mime="text/csv"
    )

    # =========================
    # ✅ CDC OFFICIAL LINK
    # =========================
    st.markdown("---")
    st.markdown(
        "🔗 [View Official CDC Clinical Guidance on Diabetes](https://www.cdc.gov/diabetes/hcp/clinical-guidance/index.html)"
    )
