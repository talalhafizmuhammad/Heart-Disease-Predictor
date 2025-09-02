import streamlit as st # type: ignore
import pandas as pd # type: ignore
import joblib as jb # type: ignore

model = jb.load('LRH.pkl')
scaler = jb.load('scaler.pkl')
Expected_cols = jb.load('columns.pkl')

st.set_page_config(page_title="Heart Disease Predictor by Talal 💊", page_icon="💊", layout="centered")

theme = st.sidebar.radio("🌓 Theme", ["Light", "Dark"])

light_css = """
<style>
.stApp { background: linear-gradient(135deg, #f7f7fb 0%, #eef2f7 100%); }
.card { background: #ffffff; border-radius: 18px; padding: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06); border: 1px solid rgba(0,0,0,0.04); }
.title { text-align:center; font-size: 2rem; font-weight: 800; color: #d6336c; margin-bottom: 6px; }
.subtitle { text-align:center; color:#6c757d; margin-bottom: 24px; }
.result { padding: 18px 20px; border-radius: 14px; font-weight: 700; text-align:center; margin-top: 12px; font-size: 1.05rem; }
.result.ok { background: #d4edda; color: #155724; }
.result.bad { background: #f8d7da; color: #721c24; }
.section { font-weight: 700; color: #334155; margin: 6px 0 10px 0; font-size: 1.05rem; }
</style>
"""

dark_css = """
<style>
.stApp { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color:#f8fafc; }
.card { background: #1e293b; border-radius: 18px; padding: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.1); }
.title { text-align:center; font-size: 2rem; font-weight: 800; color: #f43f5e; margin-bottom: 6px; }
.subtitle { text-align:center; color:#cbd5e1; margin-bottom: 24px; }
.result { padding: 18px 20px; border-radius: 14px; font-weight: 700; text-align:center; margin-top: 12px; font-size: 1.05rem; }
.result.ok { background: #14532d; color: #bbf7d0; }
.result.bad { background: #7f1d1d; color: #fecaca; }
.section { font-weight: 700; color: #f1f5f9; margin: 6px 0 10px 0; font-size: 1.05rem; }
</style>
"""

if theme == "Light":
    st.markdown(light_css, unsafe_allow_html=True)
else:
    st.markdown(dark_css, unsafe_allow_html=True)

st.markdown('<div class="title">❤️ Heart Disease Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter your reports below — model & preprocessing unchanged</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.form("predict_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section">👤 Demographics</div>', unsafe_allow_html=True)
            age = st.slider('Age', 18, 100, 30)
            sex = st.selectbox("Sex", ["Male", "Female"])

            st.markdown('<div class="section">🩺 Vitals</div>', unsafe_allow_html=True)
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
            cholesterol = st.number_input("Cholesterol level (mg/dL)", 100, 600, 200)
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

        with col2:
            st.markdown('<div class="section">💓 Cardiac</div>', unsafe_allow_html=True)
            chest_pain = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'ASY', 'TA'])
            resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
            max_hr = st.slider("Maximum Heart Rate", 80, 220, 150)
            exercise_angina = st.selectbox("Exercise-Induced Angina", ['Y', 'N'])
            old_peak = st.slider("Old Peak (ST depression)", 0.0, 6.0, 1.0)
            st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

        submitted = st.form_submit_button("🔍 Predict Now", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    raw_input = {
        'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol,
        'FastingBS': fasting_bs, 'MaxHR': max_hr, 'Oldpeak': old_peak,
        'Sex_' + sex: 1, 'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1, 'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }
    input_df = pd.DataFrame([raw_input])

    for col in Expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[Expected_cols]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.markdown('<div class="result bad">⚠️ High Risk of Heart Disease</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result ok">✅ Low Risk of Heart Disease</div>', unsafe_allow_html=True)

    with st.expander("See encoded feature row (debug)", expanded=False):
        st.dataframe(input_df, use_container_width=True)
