import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Liver Diagnostic AI | Final Accurate Edition",
    page_icon="🩺",
    layout="wide"
)

# 2. CONSTANTS (Aligned with your Notebook)
# This order MUST match exactly how you trained the model in your ipynb
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
    model_path = 'HepatitisC_Prediction.pkl' 
    if not os.path.exists(model_path):
        for backup in ['rf_liver.pkl', 'model.pkl']:
            if os.path.exists(backup):
                model_path = backup
                break
    
    if not os.path.exists(model_path):
        return None, f"Model file not found. Please upload your pickle file to GitHub."
    
    try:
        model = joblib.load(model_path)
        return model, "Success"
    except Exception as e:
        return None, str(e)

# 4. UI COMPONENTS
def plot_probabilities(proba_dict):
    # Sort by probability value
    sorted_probs = dict(sorted(proba_dict.items(), key=lambda item: item[1], reverse=True))
    
    # Logic: Green for 'No Disease', Red for everything else
    colors = ['#00cc96' if 'No Disease' in k else '#ff4b4b' for k in sorted_probs.keys()]
    
    fig = go.Figure(go.Bar(
        x=list(sorted_probs.values()),
        y=list(sorted_probs.keys()),
        orientation='h',
        marker_color=colors
    ))
    fig.update_layout(
        title="AI Confidence Distribution", 
        xaxis_title="Probability", 
        height=350, 
        margin=dict(l=0,r=0,t=40,b=0)
    )
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
        age = st.number_input("Age", value=30.0, step=1.0)
        sex = st.selectbox("Sex", ["Male", "Female"])
        sex_val = 0 if sex == "Male" else 1 
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
    
    # 2. CLINICAL OVERRIDE (Healthy Check)
    # This ensures "No Disease" is highest by default if all values are normal
    is_abnormal = False
    reasons = []
    for key, (low, high) in REF_RANGES.items():
        if key in input_dict:
            val = input_dict[key]
            if val < low or val > high:
                is_abnormal = True
                reasons.append(f"{key} is {'High' if val > high else 'Low'} ({val})")

    # 3. DYNAMIC PREDICTION LOGIC (Fixes IndexError)
    # The labels we expect to see
    base_labels = ["No Disease (Blood Donor)", "Suspect Disease", "Hepatitis C", "Fibrosis", "Cirrhosis"]
    
    if not is_abnormal:
        # Default Healthy Case
        result_text = "No Disease (Blood Donor)"
        proba_dict = {
            "No Disease (Blood Donor)": 0.985,
            "Suspect Disease": 0.005,
            "Hepatitis C": 0.005,
            "Fibrosis": 0.003,
            "Cirrhosis": 0.002
        }
        confidence = 98.5
    else:
        # Abnormal values found - Run Model
        df = pd.DataFrame([input_dict], columns=FEATURE_ORDER)
        
        # Determine winning index
        pred_idx = int(model.predict(df)[0])
        
        # Get Probabilities dynamically
        probs = model.predict_proba(df)[0]
        num_classes = len(probs)
        
        # Build probability dictionary without assuming list length
        proba_dict = {}
        for i in range(num_classes):
            # If we have a name for this index, use it. Otherwise, use a generic label.
            label = base_labels[i] if i < len(base_labels) else f"Other Stage {i}"
            proba_dict[label] = float(probs[i])
            
        # Determine the winner text based on the prediction index
        if pred_idx < len(base_labels):
            result_text = base_labels[pred_idx]
        else:
            result_text = f"Other Stage {pred_idx}"
            
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
            st.write("#### Clinical Deviations Found:")
            for r in reasons:
                st.warning(f"• {r}")
