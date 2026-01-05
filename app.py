import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="Liver Diagnostic AI", 
    page_icon="🩺", 
    layout="wide"
)

# --- CLASS MAPPING ---
CLASS_MAP = {
    0: 'No Disease (Blood Donor)',
    1: 'Suspect Disease',
    2: 'Hepatitis C',
    3: 'Fibrosis',
    4: 'Cirrhosis'
}

# --- HELPER FUNCTIONS ---
def plot_probabilities(proba_dict):
    """Creates a simple bar chart of probabilities."""
    # Strict float conversion to prevent errors
    clean_data = {k: float(v) for k, v in proba_dict.items()}
    
    # Sort for better visualization
    sorted_probs = dict(sorted(clean_data.items(), key=lambda item: item[1], reverse=True))
    
    keys = list(sorted_probs.keys())
    values = list(sorted_probs.values())
    
    # Text labels (safely formatted)
    text_labels = [f"{v*100:.1f}%" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=keys,
        orientation='h',
        text=text_labels,
        textposition='auto',
        marker_color=['#00cc96' if 'No Disease' in k else '#ff4b4b' for k in keys]
    ))
    fig.update_layout(
        title="Confidence Analysis", 
        xaxis_title="Probability", 
        height=300, 
        margin=dict(l=0,r=0,t=30,b=0)
    )
    return fig

# 2. Load Resources
@st.cache_resource
def load_resources():
    try:
        with open('rf_liver.pkl', 'rb') as f:
            model = pickle.load(f)
        try:
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
        except:
            scaler = None
        return model, scaler
    except Exception as e:
        return None, str(e)

# --- SIDEBAR ---
with st.sidebar:
    st.title("Liver AI Diagnostic")
    st.info("System Ready.")

# --- MAIN PAGE ---
st.title("🩺 Advanced Liver Disease Prediction")

# Load model
resources = load_resources()
if resources[0] is None:
    st.error("Error loading model. Make sure 'rf_liver.pkl' is in the folder.")
    st.stop()

model, scaler = resources

# INPUT FORM
with st.form("main_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Demographics")
        age = st.number_input("Age", value=45.0)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    with c2:
        st.subheader("Enzymes")
        alt = st.number_input("ALT", value=20.0)
        ast = st.number_input("AST", value=25.0)
        alp = st.number_input("ALP", value=50.0)
        ggt = st.number_input("GGT", value=20.0)
    with c3:
        st.subheader("Proteins")
        alb = st.number_input("ALB", value=38.0)
        prot = st.number_input("PROT", value=70.0)
        bil = st.number_input("BIL", value=5.0)
        che = st.number_input("CHE", value=9.0)
        chol = st.number_input("CHOL", value=4.5)
        crea = st.number_input("CREA", value=70.0)

    analyze = st.form_submit_button("Run Analysis", use_container_width=True)

if analyze:
    # 1. Prepare Input
    cols_order = ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
    input_data = [age, sex, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot]
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data], columns=cols_order)

    # Scale
    if scaler:
        final_input = scaler.transform(input_df.values)
    else:
        final_input = input_df

    try:
        # --- CRITICAL FIX SECTION ---
        # 1. Get raw prediction array
        raw_pred = model.predict(final_input)
        
        # 2. Force conversion to simple integer (removes any tuple/array wrapper)
        if hasattr(raw_pred, 'item'):
            pred_idx = int(raw_pred.item())
        else:
            pred_idx = int(raw_pred[0])
            
        result_text = CLASS_MAP.get(pred_idx, "Unknown Condition")
        
        # 3. Get probabilities
        raw_probs = model.predict_proba(final_input)
        
        # 4. Flatten ensures we have a simple 1D list of numbers
        flat_probs = raw_probs.flatten()
        
        # 5. Build dictionary with explicit float casting
        proba_dict = {CLASS_MAP[i]: float(flat_probs[i]) for i in range(len(flat_probs))}
        
        # 6. Get confidence value as a pure float
        conf_val = float(proba_dict.get(result_text, 0.0))

        # --- DISPLAY ---
        st.divider()
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if pred_idx == 0:
                st.success(f"### Diagnosis: {result_text}")
            else:
                st.error(f"### Diagnosis: {result_text}")
                
        with col2:
            # This is where your error was happening. 
            # Now 'conf_val' is guaranteed to be a float.
            st.metric("Confidence", f"{conf_val*100:.1f}%")

        # Visuals
        st.plotly_chart(plot_probabilities(proba_dict), use_container_width=True)
        
        # Debug Info (Simple)
        with st.expander("Show Technical Data"):
            st.write("Processed Input Data:")
            st.write(final_input)

    except Exception as e:
        st.error(f"An error occurred: {e}")
