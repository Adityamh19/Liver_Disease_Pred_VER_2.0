import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="Liver Diagnostic AI | Professional Edition",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER: Medical Reference Ranges (Standardized to µmol/L for Model) ---
REF_RANGES = {
    'age': (0.0, 120.0),
    'albumin': (35, 55),
    'alkaline_phosphatase': (40, 150),
    'alanine_aminotransferase': (7, 56),
    'aspartate_aminotransferase': (10, 40),
    'bilirubin': (5.0, 21.0), # Normal range in µmol/L
    'cholinesterase': (4, 12),
    'cholesterol': (2.5, 7.8),
    'creatinina': (50, 110),
    'gamma_glutamyl_transferase': (9, 48),
    'protein': (60, 80)
}

# --- CLASS MAPPING ---
CLASS_MAP = {
    0: 'No Disease (Blood Donor)',
    1: 'Suspect Disease',
    2: 'Hepatitis C',
    3: 'Fibrosis',
    4: 'Cirrhosis'
}

def get_abnormalities(inputs):
    """Identifies which markers are out of range based on the MODEL values."""
    issues = []
    for feature, value in inputs.items():
        if feature == 'sex': continue
        display_name = feature.replace('_', ' ').title()
        low, high = REF_RANGES.get(feature, (0, 9999))
        
        val = float(value)
        if val < low:
            issues.append(f"Low {display_name} ({val:.2f})")
        elif val > high:
            issues.append(f"Elevated {display_name} ({val:.2f})")
    return issues

def plot_probabilities(proba_dict):
    """Creates a professional bar chart of prediction probabilities."""
    clean_dict = {k: float(v) for k, v in proba_dict.items()}
    sorted_probs = dict(sorted(clean_dict.items(), key=lambda item: item[1], reverse=True))
    
    fig = go.Figure(go.Bar(
        x=list(sorted_probs.values()),
        y=list(sorted_probs.keys()),
        orientation='h',
        marker_color=['#00cc96' if 'No Disease' in k else '#ff4b4b' for k in sorted_probs.keys()]
    ))
    fig.update_layout(title="AI Confidence Distribution", xaxis_title="Probability", height=300, margin=dict(l=0,r=0,t=30,b=0))
    return fig

# 2. Load Resources
@st.cache_resource
def load_resources():
    model = None
    try:
        with open('rf_liver.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        return None, str(e)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3050/3050479.png", width=80)
    st.title("Liver AI Diagnostic")
    
    st.markdown("### ⚙️ Settings")
    # *** CRITICAL FIX: Unit Selection ***
    # This allows you to enter 0.8 (US) and the model receives 13.6 (Intl)
    unit_mode = st.radio("Bilirubin Units:", ["mg/dL (USA Standard)", "µmol/L (International)"])
    
    st.info("System Ready.")
    st.markdown("---")
    st.markdown("**Detectable Conditions:**")
    for v in CLASS_MAP.values():
        st.markdown(f"- {v}")

# --- MAIN PAGE ---
st.title("🩺 Advanced Liver Disease Prediction")
st.markdown("### Clinical Interface")

# Load model
resources = load_resources()
if isinstance(resources, tuple) and resources[0] is None:
    st.error(f"🚨 System Error: {resources[1]}")
    st.stop()
else:
    model = resources

# INPUT FORM
with st.form("main_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1. Demographics")
        age = st.number_input("Age (Years)", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    with c2:
        st.subheader("2. Enzymes")
        alt = st.number_input("ALT (Alanine Transaminase)", value=22.0)
        ast = st.number_input("AST (Aspartate Transaminase)", value=24.0)
        alp = st.number_input("ALP (Alkaline Phosphatase)", value=70.0) 
        ggt = st.number_input("GGT (Gamma-Glutamyl Transferase)", value=20.0)
    with c3:
        st.subheader("3. Proteins")
        alb = st.number_input("ALB (Albumin)", value=45.0)
        prot = st.number_input("PROT (Total Protein)", value=72.0)
        
        # *** SMART BILIRUBIN INPUT ***
        if "mg/dL" in unit_mode:
            # User enters US units (Default 0.8)
            bil_input = st.number_input("BIL (Bilirubin) [mg/dL]", value=0.8, step=0.1)
            # We convert it to µmol/L for the model (x 17.1)
            bil_final = bil_input * 17.1
        else:
            # User enters International units (Default 14.0)
            bil_input = st.number_input("BIL (Bilirubin) [µmol/L]", value=14.0, step=1.0)
            bil_final = bil_input

        che = st.number_input("CHE (Cholinesterase)", value=9.0)
        chol = st.number_input("CHOL (Cholesterol)", value=5.2)
        crea = st.number_input("CREA (Creatinine)", value=75.0)

    analyze = st.form_submit_button("🔍 Run Advanced Analysis", use_container_width=True)

if analyze:
    # 1. Prepare Data Dictionary (Using the FINAL converted values)
    model_input_data = {
        'Age': age, 'Sex': sex, 'ALB': alb, 'ALP': alp, 'ALT': alt, 'AST': ast,
        'BIL': bil_final, # This is now guaranteed to be in µmol/L
        'CHE': che, 'CHOL': chol, 'CREA': crea, 'GGT': ggt, 'PROT': prot
    }
    
    # Helper dict for mapping names to values for the 'Abnormalities' function
    raw_input_for_display = {
        'age': age, 'sex': sex, 'albumin': alb, 'alkaline_phosphatase': alp,
        'alanine_aminotransferase': alt, 'aspartate_aminotransferase': ast,
        'bilirubin': bil_final, 
        'cholinesterase': che, 'cholesterol': chol,
        'creatinina': crea, 'gamma_glutamyl_transferase': ggt, 'protein': prot
    }

    # 2. Prepare DataFrame
    cols_order = ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
    input_df = pd.DataFrame([model_input_data], columns=cols_order)

    try:
        # Prediction
        final_input = input_df # Direct input, no scaler
        raw_pred = model.predict(final_input)
        
        if hasattr(raw_pred, 'item'):
            pred_idx = int(raw_pred.item())
        else:
            pred_idx = int(raw_pred.flatten()[0])
            
        result_text = CLASS_MAP.get(pred_idx, "Unknown Condition")
        
        # Probabilities
        raw_probs = model.predict_proba(final_input)
        probs = raw_probs.flatten()
        proba_dict = {CLASS_MAP[i]: float(p) for i, p in enumerate(probs)}
        
        # --- RESULTS DISPLAY ---
        st.divider()
        col_res, col_conf = st.columns([3, 1])
        with col_res:
            if pred_idx == 0: 
                st.success(f"### Primary Diagnosis: {result_text}")
            else:
                st.error(f"### Primary Diagnosis: {result_text}")
        with col_conf:
            conf_val = float(proba_dict.get(result_text, 0))
            st.metric("Confidence", f"{conf_val*100:.1f}%")

        # --- TABS SECTION ---
        t1, t2, t3 = st.tabs(["📊 Confidence Analysis", "🧬 Clinical Factors", "⚙️ Debug Info"])
        
        with t1:
            st.plotly_chart(plot_probabilities(proba_dict), use_container_width=True)
            
        with t2:
            st.write("#### Deviations from Normal Range:")
            # We check the CONVERTED values against the international reference range
            abnormalities = get_abnormalities(raw_input_for_display)
            if abnormalities:
                for issue in abnormalities:
                    st.warning(f"• {issue}")
            else:
                st.success("• All biomarkers within reference range.")

        with t3:
            st.write("### Data Sent to Model")
            st.info(f"Bilirubin Mode: {unit_mode}")
            st.info(f"User Input: {bil_input} | Model Received: {bil_final:.2f} µmol/L")
            st.dataframe(input_df.style.format("{:.2f}"))

    except Exception as e:
        st.error(f"Error: {e}")
