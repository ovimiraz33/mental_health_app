# ✅ Final Streamlit App with Custom Banner & Welcome Message
import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load models
stress_model = load_model('catboost_stress_model')
anxiety_model = load_model('catboost_anxiety_model')
depression_model = load_model('catboost_depression_model')

st.set_page_config(page_title="Mental Health Screening", layout="centered")

# --- Custom Banner ---
st.markdown("""
    <div style='background-color:#f8f9fa;padding:15px;border-radius:10px'>
        <h1 style='color:#4B8BBE;text-align:center;'>🧠 Mental Health Prediction for University Students</h1>
        <h4 style='text-align:center;color:#6c757d;'>Powered by Machine Learning | Anonymous & Confidential</h4>
    </div>
""", unsafe_allow_html=True)

st.markdown("\n")
st.markdown("This web app uses trained machine learning models to assess stress, anxiety, and depression levels based on your responses. Please take a moment to fill in your details honestly.")

# --- Form for Input ---
with st.form("user_form"):
    st.subheader("📋 Demographic and Academic Details")
    age = st.selectbox("Age Range", ["<20", "20–22", "23–25", ">25"])
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    university = st.selectbox("University", ["DIU", "NSU", "BRAC", "Other"])
    department = st.text_input("Department")
    year = st.selectbox("Academic Year", ["1st", "2nd", "3rd", "4th"]) 
    cgpa = st.number_input("Current CGPA", min_value=2.0, max_value=4.0, step=0.01)
    waiver = st.selectbox("Scholarship/Waiver", ["Yes", "No"])

    st.subheader("🧪 PSS: Perceived Stress Scale (0–4)")
    pss = [st.slider(f"PSS{i}", 0, 4, 2) for i in range(1, 11)]

    st.subheader("📈 GAD: Anxiety Scale (0–3)")
    gad = [st.slider(f"GAD{i}", 0, 3, 1) for i in range(1, 8)]

    st.subheader("📉 PHQ: Depression Scale (0–3)")
    phq = [st.slider(f"PHQ{i}", 0, 3, 1) for i in range(1, 10)]

    submitted = st.form_submit_button("🔎 Predict My Mental Health")

if submitted:
    user_input = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'University': university,
        'Department': department,
        'Academic_Year': year,
        'Current_CGPA': cgpa,
        'waiver_or_scholarship': waiver,
        **{f'PSS{i+1}': pss[i] for i in range(10)},
        **{f'GAD{i+1}': gad[i] for i in range(7)},
        **{f'PHQ{i+1}': phq[i] for i in range(9)}
    }])

    stress_result = predict_model(stress_model, data=user_input)
    anxiety_result = predict_model(anxiety_model, data=user_input)
    depression_result = predict_model(depression_model, data=user_input)

    st.success("✅ Prediction Complete!")
    st.markdown("### 🧾 Results:")
    st.write("**Stress Level:**", stress_result['prediction_label'][0])
    st.write("**Anxiety Level:**", anxiety_result['prediction_label'][0])
    st.write("**Depression Level:**", depression_result['prediction_label'][0])

    st.info("These predictions are based on your answers and are not a medical diagnosis. Please reach out to a counselor if you’re concerned about your results.")
