# ══════════════════════════════════════════════════════════════════
#   Heart Disease Prediction — Streamlit App
#   Deep Learning: ANN 


import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

@st.cache_resource
def load_model_and_scaler():
    model  = pickle.load(open("heart_model_sklearn.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("❤️ Heart Disease Prediction App")
st.markdown("""
**Deep Learning — Artificial Neural Network (ANN) | TensorFlow / Keras**  
Adjust the sliders below and click **Predict** to check your heart disease risk.
""")
st.divider()

st.subheader("🔢 Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    age            = st.slider("Age (years)",               min_value=20,  max_value=90,  value=45,  step=1)
    blood_pressure = st.slider("Blood Pressure (mmHg)",     min_value=60,  max_value=200, value=120, step=1)
    cholesterol    = st.slider("Cholesterol Level (mg/dL)", min_value=100, max_value=400, value=200, step=1)

with col2:
    bmi         = st.slider("BMI",                     min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    sleep_hours = st.slider("Sleep Hours (per night)", min_value=3.0,  max_value=12.0, value=7.0,  step=0.5)

st.divider()

if st.button("🔍 Predict", use_container_width=True, type="primary"):

    input_data   = np.array([[age, blood_pressure, cholesterol, bmi, sleep_hours]])
    input_scaled = scaler.transform(input_data)
    probability  = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Prediction Result")

    if probability >= 0.5:
        st.error("⚠️ **Heart Disease Detected**")
        st.metric(label="Risk Probability", value=f"{probability * 100:.1f}%")
        st.warning("**Status: HIGH RISK**  \nPlease consult a cardiologist immediately.")
    else:
        st.success("✅ **No Heart Disease Detected**")
        st.metric(label="Risk Probability", value=f"{probability * 100:.1f}%")
        st.info("**Status: LOW RISK**  \nMaintain a healthy lifestyle and regular checkups.")

    st.markdown("**Risk Level Indicator:**")
    st.progress(float(probability))

st.divider()
st.caption("Made by IFRAH JAMIL | FA23-BBD-059 | Deep Learning Project")