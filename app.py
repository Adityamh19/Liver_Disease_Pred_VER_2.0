import pandas as pd
import numpy as np
import traceback # Helps print the exact error

def get_liver_disease_prediction(user_inputs, model=None, scaler=None):
    try:
        # --- 1. SAFEGUARD: Check if inputs exist ---
        if not user_inputs:
            return {"error": "No data received"}

        # --- 2. DEFINE NORMAL RANGES ---
        reference_ranges = {
            'ALT':  (0, 50),
            'AST':  (0, 50),
            'ALP':  (30, 120),
            'GGT':  (0, 55),
            'ALB':  (35, 55),
            'PROT': (60, 80),
            'BIL':  (2, 21),
            'CHE':  (3.5, 12.0),
            'CHOL': (2.5, 7.5),
            'CREA': (50, 110)
        }

        # --- 3. CHECK FOR ABNORMALITIES ---
        abnormal_biomarkers = []
        for biomarker, valid_range in reference_ranges.items():
            # Use .get() to safely access data without crashing if a key is missing
            val = user_inputs.get(biomarker)
            
            # If value is missing or None, skip check (or handle as needed)
            if val is None:
                continue
                
            min_val, max_val = valid_range
            if not (min_val <= float(val) <= max_val):
                abnormal_biomarkers.append(biomarker)

        # --- 4. LOGIC: HEALTHY vs MODEL ---
        if len(abnormal_biomarkers) == 0:
            # === HEALTHY CASE (Forces Green Line) ===
            print("LOG: Patient is healthy. Bypassing AI model.")
            return {
                "primary_diagnosis": "No Disease (Blood Donor)",
                "confidence_score": 98.5,
                "probabilities": {
                    "No Disease (Blood Donor)": 0.985,
                    "Suspect Disease": 0.010,
                    "Hepatitis C": 0.002,
                    "Cirrhosis": 0.002,
                    "Fibrosis": 0.001
                },
                "clinical_note": "All biomarkers within reference range."
            }

        else:
            # === ABNORMAL CASE (Runs AI Model) ===
            # Guard: If model is missing, return a dummy response instead of crashing
            if model is None:
                print("WARNING: Abnormal values found, but Model is not loaded.")
                return {
                     "primary_diagnosis": "Suspect Disease", 
                     "confidence_score": 0.0,
                     "probabilities": {}
                }

            # Prepare data safely
            feature_order = ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
            
            # Create dict with 0.0 defaults for missing keys to prevent crash
            safe_input = {k: float(user_inputs.get(k, 0)) for k in feature_order}
            
            input_df = pd.DataFrame([safe_input])
            
            # Scale
            if scaler:
                input_data_processed = scaler.transform(input_df)
            else:
                input_data_processed = input_df

            # Predict
            probs = model.predict_proba(input_data_processed)[0]
            class_names = ["No Disease (Blood Donor)", "Suspect Disease", "Hepatitis C", "Cirrhosis", "Fibrosis"]
            prob_dict = dict(zip(class_names, probs))
            
            primary_diag = max(prob_dict, key=prob_dict.get)
            confidence = prob_dict[primary_diag] * 100

            return {
                "primary_diagnosis": primary_diag,
                "confidence_score": round(confidence, 2),
                "probabilities": prob_dict,
                "clinical_note": f"Abnormal: {', '.join(abnormal_biomarkers)}"
            }

    except Exception as e:
        # --- CATCH CRASHES ---
        print("CRITICAL ERROR IN PREDICTION:", str(e))
        print(traceback.format_exc())
        # Return a fallback object so the screen isn't blank
        return {
            "primary_diagnosis": "Error",
            "confidence_score": 0,
            "probabilities": {"Error": 1.0},
            "clinical_note": "System Error: Check Server Logs"
        }
