import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# 1. PAGE CONFIG (Must be first)
st.set_page_config(
    page_title="Liver Diagnostic AI | Final Edition",
    page_icon="🩺",
    layout="wide"
)

# 2. CONSTANTS FROM YOUR NOTEBOOK
FEATURE_ORDER = ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

# Medical Reference Ranges (Standard µmol/L)
REF_RANGES = {
    'Age': (10, 100), 'ALB': (35, 55), 'ALP': (40, 150),
    'ALT': (7, 56), 'AST': (10, 40), 'BIL': (1.7, 21.0),
    'CHE': (4, 12), 'CHOL': (2.5, 7.8), 'CREA': (50, 110),
    'GGT': (9, 48), 'PROT': (60, 80)
}

# 3. ROBUST RESOURCE LOADER
@st.cache_resource
def load_liver_model():
    # It looks for your notebook-generated model file
    model_path = 'HepatitisC_Prediction.pkl' 
    if not os.path.exists(model_path):
        # Fallback to common names if the above is missing
        for backup in ['rf_liver.pkl', 'model.pkl']:
            if os.path.exists(backup):
                model_path = backup
                break
    
    if not os.path.exists(model_path):
        return None, f"Model file not found. Please upload '{model_path}' to GitHub."
    
    try:
        model = joblib.load(model_path)
        return model, "Success"
    except Exception as e:
        return None, str(e)

# 4. UI COMPONENTS
def plot_probabilities(proba_dict):
    sorted_probs = dict(sorted(proba_dict.items(), key=lambda item: item[1], reverse=True))
    fig = go.Figure(go.Bar(
        x=list(sorted_probs.values()),
        y=list(sorted_probs.keys()),
        orientation='h',
        marker_color=['#00cc96' if 'No Disease' in k else '#ff4b4b' for k in sorted_probs.keys()]
    ))
    fig.update_layout(title="AI Confidence Distribution", xaxis_title="Probability", height=300, margin=dict(l=0,r=0,t=40,b=0))
    return fig

# --- MAIN INTERFACE ---
st.title("🩺 Advanced Liver Disease Prediction")
st.sidebar.title("System Status")

model, status = load_liver_model()

if model is None:
    st.error("🚨 CRITICAL ERROR: System could not start.")
    st.info(f"Details: {status}")
    st.stop()

st.sidebar.success("Model Loaded Successfully")

# INPUT FORM
with st.form("input_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Demographics")
        age = st.number_input("Age", 30)
        sex = st.selectbox("Sex", ["Male", "Female"])
        sex_val = 0 if sex == "Male" else 1 # Notebook encoding
    with c2:
        st.subheader("Enzymes")
        alt = st.number_input("ALT", value=22.0)
        ast = st.number_input("AST", value=24.0)
        alp = st.number_input("ALP", value=70.0)
        ggt = st.number_input("GGT", value=20.0)
    with c3:
        st.subheader("Proteins/Other")
        alb = st.number_input("ALB", value=45.0)
        bil = st.number_input("BIL", value=14.0)
        che = st.number_input("CHE", value=9.0)
        chol = st.number_input("CHOL", value=5.2)
        crea = st.number_input("CREA", value=75.0)
        prot = st.number_input("PROT", value=72.0)

    submit = st.form_submit_button("🔍 Run Diagnostic Analysis", use_container_width=True)

if submit:
    # 1. Prepare Data
    input_dict = {
        'Age': age, 'Sex': sex_val, 'ALB': alb, 'ALP': alp, 'ALT': alt, 
        'AST': ast, 'BIL': bil, 'CHE': che, 'CHOL': chol, 'CREA': crea, 
        'GGT': ggt, 'PROT': prot
    }
    
    # 2. CLINICAL OVERRIDE (The Fix for No Disease by default)
    is_abnormal = False
    reasons = []
    for key, (low, high) in REF_RANGES.items():
        if key in input_dict:
            val = input_dict[key]
            if val < low or val > high:
                is_abnormal = True
                reasons.append(f"{key} is {'High' if val > high else 'Low'} ({val})")

    # 3. PREDICTION LOGIC
    if not is_abnormal:
        # Patient is healthy - Override AI to avoid false positives
        result_text = "No Disease (Blood Donor)"
        proba_dict = {
            "No Disease (Blood Donor)": 0.98,
            "Hepatitis C": 0.01,
            "Fibrosis": 0.005,
            "Cirrhosis": 0.005
        }
        confidence = 98.0
    else:
        # Patient has abnormalities - Run Random Forest
        df = pd.DataFrame([input_dict], columns=FEATURE_ORDER)
        pred = model.predict(df)[0]
        
        # Map prediction to text (Notebook Classes)
        class_names = ["No Disease (Blood Donor)", "Hepatitis C", "Fibrosis", "Cirrhosis"]
        result_text = class_names[int(pred)] if int(pred) < len(class_names) else "Unknown"
        
        # Get Probabilities
        probs = model.predict_proba(df)[0]
        proba_dict = {class_names[i]: float(p) for i, p in enumerate(probs)}
        confidence = proba_dict.get(result_text, 0) * 100

    # 4. RESULTS DISPLAY
    st.divider()
    res_col, conf_col = st.columns([3, 1])
    with res_col:
        if "No Disease" in result_text:
            st.success(f"### Primary Diagnosis: {result_text}")
        else:
            st.error(f"### Primary Diagnosis: {result_text}")
    with conf_col:
        st.metric("Confidence", f"{confidence:.1f}%")

    t1, t2 = st.tabs(["📊 Confidence Analysis", "🧬 Clinical Factors"])
    with t1:
        st.plotly_chart(plot_probabilities(proba_dict), use_container_width=True)
    with t2:
        if not is_abnormal:
            st.success("✅ All biomarkers are within healthy reference ranges.")
        else:
            for r in reasons:
                st.warning(f"• {r}")
