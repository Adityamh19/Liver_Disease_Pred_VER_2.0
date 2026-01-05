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

# --- HELPER: Medical Reference Ranges ---
REF_RANGES = {
    'age': (0.0, 120.0),
    'albumin': (35, 55),
    'alkaline_phosphatase': (40, 150),
    'alanine_aminotransferase': (7, 56),
    'aspartate_aminotransferase': (10, 40),
    'bilirubin': (1.7, 20.5),
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
    """Identifies which markers are out of range."""
    issues = []
    for feature, value in inputs.items():
        if feature == 'sex': continue
        display_name = feature.replace('_', ' ').title()
        low, high = REF_RANGES.get(feature, (0, 9999))
        
        # Ensure value is a simple float for comparison
        val_check = float(value)
        
        if val_check < low:
            issues.append(f"Low {display_name} ({val_check})")
        elif val_check > high:
            issues.append(f"Elevated {display_name} ({val_check})")
    return issues

def plot_probabilities(proba_dict):
    """
    Creates a bar chart with 'No Disease' fixed at the bottom.
    Includes strict type enforcement to prevent tuple errors.
    """
    # 1. Separate 'No Disease' from the rest
    no_disease_key = 'No Disease (Blood Donor)'
    
    # STRICT SAFETY: Ensure we extract a single float value
    raw_val = proba_dict.get(no_disease_key, 0.0)
    if isinstance(raw_val, (list, tuple, np.ndarray)):
        no_disease_prob = float(raw_val[0])
    else:
        no_disease_prob = float(raw_val)
    
    # Get all other conditions
    other_conditions = {}
    for k, v in proba_dict.items():
        if k != no_disease_key:
            # STRICT SAFETY for loop values
            if isinstance(v, (list, tuple, np.ndarray)):
                other_conditions[k] = float(v[0])
            else:
                other_conditions[k] = float(v)
    
    # 2. Sort the other conditions (Ascending order puts the highest bar at the TOP)
    sorted_others = sorted(other_conditions.items(), key=lambda item: item[1])
    
    # 3. Combine: Put No Disease first
    keys = [no_disease_key] + [k for k, v in sorted_others]
    vals = [no_disease_prob] + [v for k, v in sorted_others]
    
    # Create text labels safely by forcing float conversion again
    text_labels = [f"{float(v)*100:.1f}%" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals,
        y=keys,
        orientation='h',
        text=text_labels,
        textposition='auto',
        marker_color=['#00cc96' if 'No Disease' in k else '#ff4b4b' for k in keys]
    ))
    fig.update_layout(
        title="AI Confidence Distribution",
        xaxis_title="Probability",
        height=300,
        margin=dict(l=0,r=0,t=30,b=0)
    )
    return fig

# 2. Load Resources
@st.cache_resource
def load_resources():
    model = None
    scaler = None
    try:
        with open('rf_liver.pkl', 'rb') as f:
            model = pickle.load(f)
        try:
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
        except FileNotFoundError:
            pass # Handle warning in main app
        return model, scaler
    except Exception as e:
        return None, str(e)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3050/3050479.png", width=80)
    st.title("Liver AI Diagnostic")
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
if resources[0] is None:
    st.error(f"🚨 System Error: {resources[1]}")
    st.stop()

model, scaler = resources
if scaler is None:
    st.warning("⚠️ Scaler file not found. Predictions may be inaccurate.")

# INPUT FORM
with st.form("main_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1. Demographics")
        age = st.number_input("Age (Years)", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    with c2:
        st.subheader("2. Enzymes")
        alt = st.number_input("ALT (Alanine Transaminase)", value=20.0)
        ast = st.number_input("AST (Aspartate Transaminase)", value=25.0)
        alp = st.number_input("ALP (Alkaline Phosphatase)", value=50.0)
        ggt = st.number_input("GGT (Gamma-Glutamyl Transferase)", value=20.0)
    with c3:
        st.subheader("3. Proteins")
        alb = st.number_input("ALB (Albumin)", value=38.0)
        prot = st.number_input("PROT (Total Protein)", value=70.0)
        bil = st.number_input("BIL (Bilirubin)", value=5.0)
        che = st.number_input("CHE (Cholinesterase)", value=9.0)
        chol = st.number_input("CHOL (Cholesterol)", value=4.5)
        crea = st.number_input("CREA (Creatinine)", value=70.0)

    analyze = st.form_submit_button("🔍 Run Advanced Analysis", use_container_width=True)

if analyze:
    # 1. Prepare Data Dictionary
    raw_input = {
        'age': age, 'sex': sex, 'albumin': alb, 'alkaline_phosphatase': alp,
        'alanine_aminotransferase': alt, 'aspartate_aminotransferase': ast,
        'bilirubin': bil, 'cholinesterase': che, 'cholesterol': chol,
        'creatinina': crea, 'gamma_glutamyl_transferase': ggt, 'protein': prot
    }

    # 2. Create DataFrame for Model
    model_input_data = {
        'Age': age, 'Sex': sex, 'ALB': alb, 'ALP': alp, 'ALT': alt, 'AST': ast,
        'BIL': bil, 'CHE': che, 'CHOL': chol, 'CREA': crea, 'GGT': ggt, 'PROT': prot
    }
    
    cols_order = ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
    input_df = pd.DataFrame([model_input_data], columns=cols_order)

    # 3. Scale the Input
    if scaler:
        final_input = scaler.transform(input_df.values)
    else:
        final_input = input_df 

    try:
        # --- FIX: STRICT FLATTENING ---
        # Get raw prediction and flatten to 1D array immediately
        raw_pred = model.predict(final_input)
        pred_idx = int(np.array(raw_pred).flatten()[0]) # Force to single integer
        
        result_text = CLASS_MAP.get(pred_idx, "Unknown Condition")
        
        # Get probabilities and flatten to 1D array immediately
        raw_probs = model.predict_proba(final_input)
        probs = np.array(raw_probs).flatten() # Force to flat list of floats
        
        # Map probabilities safely
        proba_dict = {CLASS_MAP[i]: float(p) for i, p in enumerate(probs)}
        
        # --- RESULTS DISPLAY ---
        st.divider()
        col_res, col_conf = st.columns([3, 1])
        
        # Safe extraction of confidence
        raw_conf = proba_dict.get(result_text, 0.0)
        conf_val = float(raw_conf) # Strict cast to float
        
        with col_res:
            if pred_idx == 0: 
                st.success(f"### Primary Diagnosis: {result_text}")
            else:
                st.error(f"### Primary Diagnosis: {result_text}")
        with col_conf:
            # The line below is where the formatting error usually happens. 
            # We now guarantee conf_val is a float.
            st.metric("Confidence", f"{conf_val*100:.1f}%")

        # TABS
        t1, t2, t3 = st.tabs(["📊 Confidence Analysis", "🧬 Clinical Factors", "⚙️ Debug Info"])
        
        with t1:
            st.plotly_chart(plot_probabilities(proba_dict), use_container_width=True)
            
        with t2:
            st.write("#### Deviations from Normal Range:")
            abnormalities = get_abnormalities(raw_input)
            if abnormalities:
                for issue in abnormalities:
                    st.warning(f"• {issue}")
            else:
                st.success("• All biomarkers within reference range.")

        with t3:
            st.write("### Data Sent to Model")
            st.info("Values shown are exactly what you entered (with column names).")
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"Error during calculation: {e}")
        # Print detailed error to helps debug if it persists
        import traceback
        st.text(traceback.format_exc())
