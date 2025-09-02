import streamlit as st  # type: ignore
import pandas as pd     # type: ignore
import joblib as jb     # type: ignore

model = jb.load('LRH.pkl')
scaler = jb.load('scaler.pkl')
Expected_cols = jb.load('columns.pkl')

st.set_page_config(page_title="Heart Disease Predictor by Talal 💊",
                   page_icon="💊", layout="centered")

theme = st.sidebar.radio("🌓 Theme", ["Light", "Dark"])

light_css = """
<style>
.stApp { background: linear-gradient(135deg, #f7f7fb 0%, #eef2f7 100%); color: #1f2937; }
.card { background: #ffffff; border-radius: 18px; padding: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06); border: 1px solid rgba(0,0,0,0.04); }
.title { text-align:center; font-size: 2rem; font-weight: 800; color: #d6336c; margin-bottom: 6px; }
.subtitle { text-align:center; color:#495057; margin-bottom: 24px; }
.result.ok { background: #d4edda; color: #155724; padding: 14px 20px; border-radius:14px;
             text-align:center; font-weight:700; margin-top:8px; font-size:1.05rem; }
.result.bad { background: #f8d7da; color: #721c24; padding: 14px 20px; border-radius:14px;
              text-align:center; font-weight:700; margin-top:8px; font-size:1.05rem; }
.section { font-weight: 700; color: #334155; margin: 6px 0 4px 0; font-size: 1.05rem; }
div.row-widget.stSlider > div, div.row-widget.stNumberInput > div, div.row-widget.stSelectbox > div { margin-top:0px; margin-bottom:6px; }

/* Force label colors in light mode */
.stSlider label, .stNumberInput label, .stSelectbox label, 
.stRadio label, div[data-baseweb="form-control"] label,
div.stSlider > label, div.stNumberInput > label, 
div.stSelectbox > label, .stMarkdown p,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
.st-emotion-cache-16idsys p, .st-emotion-cache-1gulkj5,
div[data-testid="column"] label {
    color: #1f2937 !important;
    font-weight: 600 !important;
}
</style>
"""

dark_css = """
<style>
.stApp { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color: #f1f5f9; }
.card { background: #1e293b; border-radius: 18px; padding: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.1); }
.title { text-align:center; font-size: 2rem; font-weight: 800; color: #f43f5e; margin-bottom: 6px; }
.subtitle { text-align:center; color:#cbd5e1; margin-bottom: 24px; }
.result.ok { background: #14532d; color: #bbf7d0; padding: 14px 20px; border-radius:14px;
             text-align:center; font-weight:700; margin-top:8px; font-size:1.05rem; }
.result.bad { background: #7f1d1d; color: #fecaca; padding: 14px 20px; border-radius:14px;
              text-align:center; font-weight:700; margin-top:8px; font-size:1.05rem; }
.section { font-weight: 700; color: #f1f5f9; margin: 6px 0 4px 0; font-size: 1.05rem; }
div.row-widget.stSlider > div, div.row-widget.stNumberInput > div, div.row-widget.stSelectbox > div { margin-top:0px; margin-bottom:6px; }

/* Force label colors in dark mode */
.stSlider label, .stNumberInput label, .stSelectbox label, 
.stRadio label, div[data-baseweb="form-control"] label,
div.stSlider > label, div.stNumberInput > label, 
div.stSelectbox > label, .stMarkdown p,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
.st-emotion-cache-16idsys p, .st-emotion-cache-1gulkj5,
div[data-testid="column"] label {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
}
</style>
"""

st.markdown(dark_css if theme=="Dark" else light_css, unsafe_allow_html=True)

label_override_css = f"""
<style>
/* Target all possible label selectors */
.stSlider > label > div > div > div > p,
.stNumberInput > label > div > div > div > p,
.stSelectbox > label > div > div > div > p,
[data-testid="stWidgetLabel"] > p,
[data-testid="stWidgetLabel"] > div > p,
.st-emotion-cache-16idsys > p,
label[data-baseweb="form-control-label"],
div[data-testid="column"] p,
div.row-widget label p,
div.stSlider p, div.stNumberInput p, div.stSelectbox p {{
    color: {"#1f2937" if theme=="Light" else "#f1f5f9"} !important;
    font-weight: 600 !important;
}}

/* Target Streamlit's emotion CSS classes */
[class*="st-emotion-cache"] p {{
    color: {"#1f2937" if theme=="Light" else "#f1f5f9"} !important;
}}

/* More specific targeting for form elements */
.stForm label, .stForm p, form label, form p {{
    color: {"#1f2937" if theme=="Light" else "#f1f5f9"} !important;
}}
</style>
"""
st.markdown(label_override_css, unsafe_allow_html=True)

st.markdown('<div class="title">❤️ Heart Disease Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter your reports below — model & preprocessing unchanged</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.form("predict_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section">👤 Demographics</div>', unsafe_allow_html=True)
            age = st.slider("Age", 18, 100, 30)
            sex = st.selectbox("Sex", ["Male", "Female"])

            st.markdown('<div class="section">🩺 Vitals</div>', unsafe_allow_html=True)
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
            cholesterol = st.number_input("Cholesterol level (mg/dL)", 100, 600, 200)
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

        with col2:
            st.markdown('<div class="section">💓 Cardiac</div>', unsafe_allow_html=True)
            chest_pain = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'ASY', 'TA'])
            resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
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

    st.markdown(f"""
    <div class="result {'bad' if prediction else 'ok'}">
        {'⚠️ High Risk of Heart Disease' if prediction else '✅ Low Risk of Heart Disease'}
    </div>
    """, unsafe_allow_html=True)

    with st.expander("See encoded feature row (debug)", expanded=False):
        st.dataframe(input_df, use_container_width=True)
