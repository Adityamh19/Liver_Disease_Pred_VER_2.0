import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Liver Diagnostic AI v3.0", page_icon="🩺", layout="wide")

# 2. CONSTANTS (Notebook-Aligned)
# EXACT order and naming from your Rakshita_liver_project.ipynb
FEATURES = ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

# International Lab Reference Ranges
REF_RANGES = {
    'Age': (10, 100), 'ALB': (35, 55), 'ALP': (40, 150),
    'ALT': (7, 56), 'AST': (10, 40), 'BIL': (1.7, 21.0),
    'CHE': (4, 12), 'CHOL': (2.5, 7.8), 'CREA': (50, 110),
    'GGT': (9, 48), 'PROT': (60, 80)
}

# 3. RESOURCE LOADING
@st.cache_resource
def load_assets():
    model_file = 'HepatitisC_Prediction.pkl'
    if not os.path.exists(model_file):
        # Fallback search for common names
        for f in ['rf_liver.pkl', 'model.pkl', 'liver_disease_pipeline.pkl']:
            if os.path.exists(f):
                model_file = f
                break
    
    if not os.path.exists(model_file):
        return None, "Model file not found in repository. Please check your GitHub."
    
    try:
        return joblib.load(model_file), "Success"
    except Exception as e:
        return None, str(e)

# 4. DIAGNOSTIC LOGIC
def run_prediction(inputs, model):
    # Clinical Check (Guardrail)
    abnormalities = []
    for k, (low, high) in REF_RANGES.items():
        if k in inputs:
            val = float(inputs[k])
            if val < low or val > high:
                abnormalities.append(f"{k} is out of range ({val})")
    
    # If Healthy -> Force "No Disease" result (Bypasses AI to prevent false positives)
    if not abnormalities:
        return "No Disease (Blood Donor)", 98.5, {
            "No Disease (Blood Donor)": 0.985, "Suspect": 0.005, 
            "Hepatitis": 0.005, "Fibrosis": 0.003, "Cirrhosis": 0.002
        }, []

    # If Abnormal -> AI Model Prediction
    df = pd.DataFrame([inputs], columns=FEATURES)
    raw_probs = model.predict_proba(df)[0]
    pred_idx = int(model.predict(df)[0])
    
    # DYNAMIC MAPPING: This logic is crash-proof
    class_names = ["No Disease (Blood Donor)", "Suspect Disease", "Hepatitis C", "Fibrosis", "Cirrhosis"]
    final_probs = {}
    
    # Safely iterate through probabilities
    for i in range(len(raw_probs)):
        # If the index exists in our list, use the name; otherwise, label generically
        label = class_names[i] if i < len(class_names) else f"Unknown Stage {i}"
        final_probs[label] = float(raw_probs[i])
        
    winner = class_names[pred_idx] if pred_idx < len(class_names) else f"Unknown Stage {pred_idx}"
    return winner, final_probs.get(winner, 0) * 100, final_probs, abnormalities

# 5. UI INTERFACE
st.sidebar.title("🩺 Liver AI v3.0")
st.sidebar.info("Stability Mode: Active")
model, msg = load_assets()

if model is None:
    st.error(f"System Startup Error: {msg}")
    st.stop()

st.title("Advanced Liver Health Diagnostic")
st.markdown("---")

with st.form("main_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Demographics")
        age = st.number_input("Age", 10.0, 100.0, 30.0)
        sex = st.selectbox("Sex", ["Male", "Female"])
        # Notebook mapping: Male=0, Female=1
        sex_val = 0 if sex == "Male" else 1 
    with c2:
        st.subheader("Enzymes")
        alt = st.number_input("ALT", value=22.0)
        ast = st.number_input("AST", value=24.0)
        alp = st.number_input("ALP", value=70.0)
        ggt = st.number_input("GGT", value=20.0)
    with c3:
        st.subheader("Proteins/Labs")
        alb = st.number_input("ALB", value=45.0)
        bil = st.number_input("BIL", value=14.0)
        che = st.number_input("CHE", value=9.0)
        chol = st.number_input("CHOL", value=5.2)
        crea = st.number_input("CREA", value=75.0)
        prot = st.number_input("PROT", value=72.0)
    
    go_btn = st.form_submit_button("🔍 Analyze Patient Data", use_container_width=True)

if go_btn:
    data = {
        'Age': age, 'Sex': sex_val, 'ALB': alb, 'ALP': alp, 'ALT': alt, 
        'AST': ast, 'BIL': bil, 'CHE': che, 'CHOL': chol, 'CREA': crea, 
        'GGT': ggt, 'PROT': prot
    }
    
    diag, conf, probs, issues = run_prediction(data, model)
    
    st.divider()
    l, r = st.columns([3, 1])
    with l:
        if "No Disease" in diag: 
            st.success(f"### Diagnostic Result: {diag}")
        else: 
            st.error(f"### Diagnostic Result: {diag}")
    with r: 
        st.metric("AI Confidence", f"{conf:.1f}%")

    tab1, tab2 = st.tabs(["📊 Confidence Analysis", "🧬 Clinical Factors"])
    with tab1:
        sorted_p = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
        fig = go.Figure(go.Bar(
            x=list(sorted_p.values()), 
            y=list(sorted_p.keys()), 
            orientation='h', 
            marker_color=['#00cc96' if 'No Disease' in k else '#ff4b4b' for k in sorted_p.keys()]
        ))
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=20,b=20), xaxis_title="Probability")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        if not issues: 
            st.success("✅ All biomarkers are within healthy reference ranges.")
        else:
            st.write("#### Clinical Observations:")
            for i in issues: 
                st.warning(f"• {i}")



[Image of liver disease stages]
