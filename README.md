# 🧠 Mental Health Screening App (Stress, Anxiety, Depression)

This Streamlit app predicts the **stress**, **anxiety**, and **depression** levels of university students using machine learning models trained on PSS, GAD-7, and PHQ-9 scores.

## 🚀 Features
- Predicts mental health condition based on survey inputs
- Uses CatBoost models trained with PyCaret
- Fully anonymous and student-friendly interface
- Real-time output with personalized predictions

## 📂 Files
- `streamlit_mental_health_app.py`: Main Streamlit application
- `catboost_stress_model.pkl`: Trained model for stress
- `catboost_anxiety_model.pkl`: Trained model for anxiety
- `catboost_depression_model.pkl`: Trained model for depression
- `requirements.txt`: Python dependencies

## 💻 How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/mental-health-app.git
cd mental-health-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run streamlit_mental_health_app.py
```

## 🌐 Deploy on Streamlit Cloud
Just connect this repo to [streamlit.io/cloud](https://streamlit.io/cloud) and click **Deploy**!

## ⚠ Disclaimer
This is a predictive tool, not a diagnostic one. Always consult a licensed professional for medical or psychological guidance.
