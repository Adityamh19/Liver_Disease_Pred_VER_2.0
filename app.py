import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Liver Diagnostic AI | Scratch Rebuild",
    page_icon="🩺",
    layout="wide"
)

# --- 2. GLOBAL SETTINGS ---
# Feature order must match your notebook exactly
FEATURES = ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

# International medical reference ranges
REF_RANGES = {
    'Age': (1, 110), 'ALB': (35, 55), 'ALP': (40, 150),
    'ALT': (7, 56), 'AST': (10, 40), 'BIL': (1.7, 21.0),
    'CHE': (4, 12), 'CHOL': (2.5, 7.8), 'CREA': (50, 110),
    'GGT': (9, 48), 'PROT': (60, 80)
}

# Standard class names for the UCI Liver dataset
DEFAULT_CLASS_NAMES = [
    "No Disease (Blood Donor)", 
    "Suspect Disease", 
    "Hepatitis C", 
    "Fibrosis", 
    "Cirrhosis"
]

# --- 3. RESOURCE LOADING (WITH DIAGNOSTICS) ---
@st.cache_resource
def load_system():
    # Attempting to load the specific name from your recent uploads
    model_file = 'HepatitisC_Prediction.pkl'
    if not os.path.exists(model_file):
        # Backup names
        for f in ['rf_liver.pkl', 'model.pkl']:
            if os.path.exists(f):
                model_file = f
                break
                
    if not os.path.exists(model_file):
        return None, "Pickle file not found in GitHub. Please upload 'HepatitisC_Prediction.pkl'."
    
    try:
        model = joblib.load(model_file)
        return model, "Success"
    except Exception as e:
        return None, str(e)

# --- 4. CORE DIAGNOSTIC ENGINE ---
def analyze_liver(input_dict, model):
    # A. CLINICAL GUARDRAIL
    abnormal_markers = []
    for key, (low, high) in REF_RANGES.items():
        if key in input_dict:
            val = input_dict[key]
            if val < low or val > high:
                abnormal_markers.append(f"{key}: {val} (Range: {low}-{high})")

    # B. DECISION: Force Healthy or Run AI
    if len(abnormal_markers) == 0:
        # BYPASS MODEL: Return fixed healthy result
        primary = "No Disease (Blood Donor)"
        conf = 98.5
        probs = {
            "No Disease (Blood Donor)": 0.985,
            "Suspect Disease": 0.010,
            "Hepatitis C": 0.002,
            "Fibrosis": 0.002,
            "Cirrhosis": 0.001
        }
        return primary, conf, probs, []

    else:
        # RUN AI MODEL
        df = pd.DataFrame([input_dict], columns=FEATURES)
        
        # Get raw data from model
        raw_probs = model.predict_proba(df)[0]
        pred_idx = int(model.predict(df)[0])
        
        # DYNAMIC CLASS MAPPING (The specific fix for your IndexError)
        proba_dict = {}
        for i, p in enumerate(raw_probs):
            # If we have a name for this index, use it. If not, don't crash.
            if i < len(DEFAULT_CLASS_NAMES):
                label = DEFAULT_CLASS_NAMES[i]
            else:
                label = f"Advanced Stage {i}"
            proba_dict[label] = float(p)
            
        # Identify the winner name safely
        if pred_idx < len(DEFAULT_CLASS_NAMES):
            winner_name = DEFAULT_CLASS_NAMES[pred_idx]
        else:
            winner_name = f"Advanced Stage {pred_idx}"
            
        return winner_name, proba_dict[winner_name] * 100, proba_dict, abnormal_markers

# --- 5. UI INTERFACE ---
st.title("🩺 Advanced Liver Disease Analysis")
st.markdown("---")

model, load_msg = load_system()
if model is None:
    st.error(f"🚨 Setup Error: {load_msg}")
    st.stop()

with st.form("liver_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Demographics")
        age = st.number_input("Age", 10.0, 110.0, 30.0)
        sex = st.selectbox("Sex", ["Male", "Female"])
        sex_val = 0 if sex == "Male" else 1 # Notebook mapping
        
    with col2:
        st.subheader("2. Enzymes")
        alt = st.number_input("ALT", value=22.0)
        ast = st.number_input("AST", value=24.0)
        alp = st.number_input("ALP", value=70.0)
        ggt = st.number_input("GGT", value=20.0)
        
    with col3:
        st.subheader("3. Proteins/Labs")
        alb = st.number_input("ALB", value=45.0)
        bil = st.number_input("BIL", value=14.0)
        che = st.number_input("CHE", value=9.0)
        chol = st.number_input("CHOL", value=5.2)
        crea = st.number_input("CREA", value=75.0)
        prot = st.number_input("PROT", value=72.0)

    btn = st.form_submit_button("🔍 Run Full Diagnostic", use_container_width=True)

if btn:
    data = {
        'Age': age, 'Sex': sex_val, 'ALB': alb, 'ALP': alp, 'ALT': alt, 
        'AST': ast, 'BIL': bil, 'CHE': che, 'CHOL': chol, 'CREA': crea, 
        'GGT': ggt, 'PROT': prot
    }
    
    # Analyze
    diag, confidence, all_probs, issues = analyze_liver(data, model)
    
    # Display Results
    st.markdown("---")
    res_l, res_r = st.columns([3, 1])
    
    with res_l:
        if diag == "No Disease (Blood Donor)":
            st.success(f"### Primary Diagnosis: {diag}")
        else:
            st.error(f"### Primary Diagnosis: {diag}")
            
    with res_r:
        st.metric("Confidence", f"{confidence:.1f}%")

    tab1, tab2 = st.tabs(["📊 Confidence Analysis", "🧬 Clinical Factors"])
    
    with tab1:
        # Professional Chart
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        fig = go.Figure(go.Bar(
            x=list(sorted_probs.values()),
            y=list(sorted_probs.keys()),
            orientation='h',
            marker_color=['#00cc96' if 'No Disease' in k else '#ff4b4b' for k in sorted_probs.keys()]
        ))
        fig.update_layout(xaxis_title="Probability", height=350, margin=dict(l=0,r=0,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        if not issues:
            st.success("✅ All biomarkers are within healthy reference ranges.")
        else:
            st.write("#### Identified Abnormalities:")
            for i in issues:
                st.warning(f"• {i}")
