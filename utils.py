"""
Utility functions for model loading, prediction, feature coefficient extraction, and report export.
"""

from pathlib import Path
import datetime
import pandas as pd
import numpy as np
import joblib
import streamlit as st

MODEL_PATH = Path(__file__).parent / "food_delivery_linear_regression.pkl"
TEST_PREDS_PATH = Path(__file__).parent / "test_predictions.csv"

EXPECTED_COLUMNS = [
    "Delivery_Person_Age",
    "Delivery_Person_Rating",
    "Weather",
    "Traffic_Level",
    "Vehicle_Type",
    "Area_Type",
    "Order_Type",
    "Festival",
    "Multiple_Deliveries",
    "Distance_km"
]

@st.cache_resource
def load_model():
    """
    Loads and validates the pre-trained Scikit-Learn pipeline.
    Returns (model, None) on success, or (None, error_message) on failure.
    """
    if not MODEL_PATH.exists():
        return None, f"Model file not found. Expected file at: '{MODEL_PATH.name}'. Please place 'food_delivery_linear_regression.pkl' in the project directory."
    
    try:
        model = joblib.load(MODEL_PATH)
        if not hasattr(model, "predict"):
            return None, "Loaded model object does not support the '.predict()' method."
        if not hasattr(model, "named_steps") and not hasattr(model, "steps"):
            return None, "Loaded model object is not a valid Scikit-Learn Pipeline."
        return model, None
    except Exception as e:
        return None, f"Model initialization failed: {str(e)}. Ensure scikit-learn==1.6.1 is installed."

def predict_delivery_time(model, input_data: dict) -> float:
    """
    Constructs a single-row DataFrame with exact expected feature names
    and executes pipeline prediction.
    """
    df = pd.DataFrame([input_data])[EXPECTED_COLUMNS]
    prediction = model.predict(df)[0]
    return float(prediction)

def format_duration(minutes: float) -> str:
    """
    Formats minute float into a friendly duration string.
    Example: 77.4 -> 'Approximately 1 hour 17 minutes'
    """
    rounded_mins = int(round(minutes))
    if rounded_mins < 60:
        return f"Approximately {rounded_mins} minute{'s' if rounded_mins != 1 else ''}"
    hours = rounded_mins // 60
    rem_mins = rounded_mins % 60
    if rem_mins == 0:
        return f"Approximately {hours} hour{'s' if hours != 1 else ''}"
    return f"Approximately {hours} hour{'s' if hours != 1 else ''} {rem_mins} minute{'s' if rem_mins != 1 else ''}"

def get_interpretation_badge(minutes: float) -> tuple[str, str]:
    """
    Returns (category label, status color) for simple demonstration ranges.
    0-30: Fast delivery
    31-60: Moderate delivery time
    61-90: Longer delivery time
    > 90: High delivery time
    """
    m = round(minutes)
    if m <= 30:
        return "Fast delivery", "#16A34A"  # Green
    elif m <= 60:
        return "Moderate delivery time", "#2563EB"  # Blue
    elif m <= 90:
        return "Longer delivery time", "#D97706"  # Amber/Orange
    else:
        return "High delivery time", "#DC2626"  # Red

def extract_coefficients(model) -> pd.DataFrame:
    """
    Dynamically extracts feature names and fitted coefficients from the pipeline.
    """
    try:
        preprocessor = model.named_steps["preprocessor"]
        regressor = model.named_steps["model"]
        
        feature_names = preprocessor.get_feature_names_out()
        coefs = regressor.coef_
        
        clean_names = []
        for name in feature_names:
            c_name = name.replace("num__", "").replace("cat__", "").replace("_", " ")
            clean_names.append(c_name)
            
        df_coef = pd.DataFrame({
            "Feature": clean_names,
            "Raw_Feature": feature_names,
            "Coefficient": coefs,
            "Abs_Coefficient": np.abs(coefs)
        })
        
        df_coef = df_coef.sort_values(by="Abs_Coefficient", ascending=False).reset_index(drop=True)
        return df_coef
    except Exception as e:
        st.warning(f"Could not extract coefficients automatically: {e}")
        return pd.DataFrame()

def load_test_predictions() -> pd.DataFrame:
    """
    Loads test set predictions if available for Actual vs Predicted visualization.
    """
    if TEST_PREDS_PATH.exists():
        try:
            return pd.read_csv(TEST_PREDS_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def generate_csv_report(input_data: dict, pred_minutes: float) -> bytes:
    """
    Generates downloadable CSV bytes summarizing the prediction request.
    """
    report_dict = {
        "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        **{k: [v] for k, v in input_data.items()},
        "Predicted_Delivery_Time_min": [round(pred_minutes, 2)],
        "Formatted_Duration": [format_duration(pred_minutes)]
    }
    df_report = pd.DataFrame(report_dict)
    return df_report.to_csv(index=False).encode('utf-8')
