import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from pathlib import Path

# Local imports
from model_results import MODEL_RESULTS, PROJECT_METADATA, FEATURE_RANGES
from utils import (
    load_model,
    predict_delivery_time,
    format_duration,
    get_interpretation_badge,
    extract_coefficients,
    load_test_predictions,
    generate_csv_report
)

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Food Delivery Time Prediction | ML Dashboard",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Dark Green, White & Black High-Contrast Design System
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@600;700;800&display=swap');
    
    /* Base Color Tokens (Dark Green, White, Black) */
    :root {
        --green-primary: #064E3B;
        --green-dark: #022C22;
        --green-gradient: linear-gradient(135deg, #064E3B 0%, #022C22 100%);
        --maroon-primary: var(--green-primary);
        --maroon-dark: var(--green-dark);
        --maroon-gradient: var(--green-gradient);
        --black-main: #0A0A0A;
        --black-soft: #1A1A1A;
        --white-pure: #FFFFFF;
        --white-off: #FAFAFA;
        --gray-border: #E5E7EB;
        --gray-text: #374151;
        --gray-muted: #6B7280;
    }
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        color: var(--black-main);
        background-color: var(--white-off);
    }
    
    .stApp {
        background-color: var(--white-off);
    }
    
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }
    
    /* Sidebar Styling - Deep Black & Dark Green Accent */
    section[data-testid="stSidebar"] {
        background-color: var(--black-main) !important;
        border-right: 2px solid var(--green-primary) !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: var(--white-pure) !important;
    }
    
    /* Header Card - Deep Dark Green Hero */
    .header-hero-card {
        background: var(--green-gradient);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        border: 2px solid var(--black-main);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        margin-bottom: 2rem;
        color: var(--white-pure);
    }
    
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        color: var(--white-pure);
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-size: 1.15rem;
        color: #F3F4F6;
        font-weight: 500;
        margin-bottom: 1.2rem;
    }
    
    .badge-green, .badge-maroon {
        display: inline-block;
        background: var(--black-main);
        color: var(--white-pure);
        border: 1px solid var(--white-pure);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 8px;
        margin-bottom: 6px;
    }

    .badge-white {
        display: inline-block;
        background: var(--white-pure);
        color: var(--green-primary);
        border: 1px solid var(--black-main);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 800;
        margin-right: 8px;
        margin-bottom: 6px;
    }

    /* KPI Cards - Crisp White with Dark Green Accent */
    .kpi-card-clean {
        background: var(--white-pure);
        border: 2px solid var(--black-main);
        border-top: 5px solid var(--green-primary);
        border-radius: 14px;
        padding: 1.4rem 1rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    
    .kpi-card-clean:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(6, 78, 59, 0.25);
    }
    
    .kpi-icon {
        font-size: 1.8rem;
        margin-bottom: 0.3rem;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 800;
        color: var(--black-main);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        color: var(--green-primary);
        margin-top: 0.2rem;
        line-height: 1.1;
    }

    /* Section Containers */
    .content-box {
        background: var(--white-pure);
        border: 2px solid var(--black-main);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.8rem;
    }

    .content-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.45rem;
        font-weight: 800;
        color: var(--black-main);
        border-bottom: 3px solid var(--green-primary);
        padding-bottom: 0.5rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Process Workflow Nodes */
    .wf-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
        margin-top: 1rem;
    }

    .wf-node-item {
        background: var(--black-main);
        color: var(--white-pure);
        border: 2px solid var(--green-primary);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }

    .wf-step-num {
        font-size: 0.75rem;
        font-weight: 800;
        color: var(--white-pure);
        background: var(--green-primary);
        padding: 2px 8px;
        border-radius: 9999px;
        display: inline-block;
        margin-bottom: 6px;
    }

    .wf-node-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--white-pure);
    }

    /* Prediction Outcome Hero Box */
    .prediction-green-hero, .prediction-maroon-hero {
        background: var(--green-gradient);
        border: 3px solid var(--black-main);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        color: var(--white-pure);
        box-shadow: 0 12px 30px rgba(6, 78, 59, 0.35);
        margin: 1.5rem 0;
    }

    .pred-hero-label {
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #F3F4F6;
    }

    .pred-hero-number {
        font-family: 'Outfit', sans-serif;
        font-size: 4.5rem;
        font-weight: 800;
        color: var(--white-pure);
        line-height: 1;
        margin: 0.4rem 0;
        text-shadow: 2px 2px 0px var(--black-main);
    }

    /* High Contrast Buttons */
    div.stButton > button {
        background: var(--green-primary) !important;
        color: var(--white-pure) !important;
        border: 2px solid var(--black-main) !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }

    div.stButton > button:hover {
        background: var(--black-main) !important;
        color: var(--white-pure) !important;
        border-color: var(--green-primary) !important;
        transform: translateY(-2px) !important;
    }

    /* Download Button */
    div.stDownloadButton > button {
        background: var(--black-main) !important;
        color: var(--white-pure) !important;
        border: 2px solid var(--green-primary) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.8rem !important;
        font-weight: 800 !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }

    div.stDownloadButton > button:hover {
        background: var(--green-primary) !important;
        border-color: var(--black-main) !important;
        transform: translateY(-2px) !important;
    }

    /* Mobile & Touch Responsive Optimization */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        .header-hero-card {
            padding: 1.4rem 1.2rem !important;
            border-radius: 12px !important;
            margin-bottom: 1.2rem !important;
        }
        .hero-title {
            font-size: 1.8rem !important;
        }
        .hero-subtitle {
            font-size: 0.98rem !important;
            margin-bottom: 0.8rem !important;
        }
        .badge-green, .badge-maroon, .badge-white {
            font-size: 0.75rem !important;
            padding: 4px 10px !important;
            margin-right: 4px !important;
            margin-bottom: 4px !important;
        }
        .kpi-card-clean {
            padding: 1rem 0.6rem !important;
            margin-bottom: 0.8rem !important;
        }
        .kpi-value {
            font-size: 1.6rem !important;
        }
        .prediction-green-hero, .prediction-maroon-hero {
            padding: 1.5rem 1rem !important;
            border-radius: 14px !important;
            margin: 1rem 0 !important;
        }
        .pred-hero-number {
            font-size: 3rem !important;
        }
        .content-box {
            padding: 1.2rem 1rem !important;
            border-radius: 12px !important;
            margin-bottom: 1.2rem !important;
        }
        .content-header {
            font-size: 1.2rem !important;
        }
        .wf-grid {
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)) !important;
            gap: 8px !important;
        }
        .wf-node-item {
            padding: 0.8rem 0.4rem !important;
        }
        .wf-node-title {
            font-size: 0.88rem !important;
        }
        div.stButton > button, div.stDownloadButton > button {
            padding: 0.8rem 1.2rem !important;
            font-size: 1rem !important;
            min-height: 48px !important;
        }
        [data-testid="stDataFrame"] {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }
    }

    @media (max-width: 480px) {
        .hero-title {
            font-size: 1.45rem !important;
        }
        .pred-hero-number {
            font-size: 2.3rem !important;
        }
        .wf-grid {
            grid-template-columns: 1fr 1fr !important;
        }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Navigation (Black & Dark Green Theme)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.2rem 0 0.8rem 0;">
        <div style="font-size: 3.2rem;">🍕</div>
        <div style="font-family: 'Outfit', sans-serif; font-size: 1.4rem; font-weight: 800; color: #FFFFFF;">Food Express ML</div>
        <div style="font-size: 0.85rem; color: #E5E7EB; font-weight: 700; letter-spacing: 0.05em;">DELIVERY DURATION SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "SELECT PAGE",
        [
            "🏠 Overview",
            "🎯 Delivery Time Prediction",
            "📊 Model Performance",
            "🔍 Feature Analysis",
            "📋 Dataset & Methodology",
            "👨‍💻 About the Project"
        ],
        index=0
    )
    
    st.markdown("---")
    
    # Model Health Status Box
    st.markdown("""
    <div style="background: #1A1A1A; border-radius: 12px; padding: 1.2rem; border: 2px solid #064E3B; font-size: 0.85rem;">
        <div style="font-weight: 800; color: #FFFFFF; margin-bottom: 4px;">⚡ System Status</div>
        <div style="color: #4ADE80; font-weight: 800; display:flex; align-items:center; gap:6px;">
            <span style="font-size:1.2rem;">●</span> Model Loaded Cleanly
        </div>
        <div style="color: #E5E7EB; margin-top: 6px;">Pipeline: Linear Regression</div>
        <div style="color: #E5E7EB;">Scikit-learn: v1.6.1</div>
    </div>
    
    <div style="background: #1A1A1A; border-radius: 12px; padding: 1rem; border: 2px solid #064E3B; font-size: 0.85rem; margin-top: 12px; text-align: center;">
        <div style="font-weight: 800; color: #FFFFFF; margin-bottom: 4px;">👨‍💻 Developer GitHub</div>
        <a href="https://github.com/Abrar-Codes404" target="_blank" style="color: #4ADE80; font-weight: 800; text-decoration: none; font-size: 0.95rem;">
            🐙 @Abrar-Codes404
        </a>
    </div>
    """, unsafe_allow_html=True)

# Load trained pipeline with caching & error validation
model, load_error = load_model()

if load_error:
    st.error("⚠️ Model Loading Error")
    st.error(load_error)
    st.info("Please ensure that `food_delivery_linear_regression.pkl` is present in the workspace root directory and compatible with scikit-learn==1.6.1.")
    st.stop()

# ---------------------------------------------------------
# Plotly Dark Green & White High-Contrast Theme Config
# ---------------------------------------------------------
PLOTLY_GREEN_THEME = dict(
    paper_bgcolor='#FFFFFF',
    plot_bgcolor='#FAFAFA',
    font=dict(family="Plus Jakarta Sans", color="#0A0A0A", size=13),
    xaxis=dict(gridcolor='#E5E7EB', zerolinecolor='#0A0A0A', showline=True, linecolor='#0A0A0A'),
    yaxis=dict(gridcolor='#E5E7EB', zerolinecolor='#0A0A0A', showline=True, linecolor='#0A0A0A')
)
PLOTLY_MAROON_THEME = PLOTLY_GREEN_THEME

# ---------------------------------------------------------
# 1. OVERVIEW PAGE
# ---------------------------------------------------------
if page == "🏠 Overview":
    st.markdown("""
    <div class="header-hero-card">
        <div class="hero-title">Food Delivery Time Prediction</div>
        <div class="hero-subtitle">Machine Learning Based Delivery Duration Estimation System</div>
        <div>
            <span class="badge-white">🍕 Food Delivery ML</span>
            <span class="badge-green">⚡ Linear Regression Pipeline</span>
            <span class="badge-green">📊 Best R²: 0.65</span>
            <span class="badge-green">🎯 Best MAE: 8.49 min</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Intro Card
    st.markdown("""
    <div class="content-box">
        <div class="content-header">🚀 Project Overview</div>
        <p style="color: #1F2937; font-size: 1.08rem; line-height: 1.7; margin: 0; font-weight: 500;">
            This machine learning application estimates food delivery duration in minutes by analyzing 
            <strong style="color: #064E3B; font-weight: 800;">10 operational variables</strong> including courier characteristics, weather severity, 
            traffic congestion, vehicle type, area classification, order category, and physical distance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Stat Grid
    st.subheader("📌 Key Performance Indicators")
    k1, k2, k3, k4, k5 = st.columns(5)
    
    with k1:
        st.markdown("""
        <div class="kpi-card-clean">
            <div class="kpi-icon">📊</div>
            <div class="kpi-label">Dataset Records</div>
            <div class="kpi-value">794</div>
            <div style="font-size: 0.78rem; color: #6B7280; font-weight: 700;">Cleaned Samples</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        st.markdown("""
        <div class="kpi-card-clean">
            <div class="kpi-icon">🎛️</div>
            <div class="kpi-label">Input Features</div>
            <div class="kpi-value" style="color: #0A0A0A;">10</div>
            <div style="font-size: 0.78rem; color: #6B7280; font-weight: 700;">4 Num + 6 Cat</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown("""
        <div class="kpi-card-clean">
            <div class="kpi-icon">⏱️</div>
            <div class="kpi-label">Target Variable</div>
            <div class="kpi-value" style="font-size: 1.4rem; color: #0A0A0A; margin-top: 0.4rem;">Duration</div>
            <div style="font-size: 0.78rem; color: #6B7280; font-weight: 700;">Minutes (Continuous)</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown("""
        <div class="kpi-card-clean">
            <div class="kpi-icon">⭐</div>
            <div class="kpi-label">Best R² Score</div>
            <div class="kpi-value">0.65</div>
            <div style="font-size: 0.78rem; color: #064E3B; font-weight: 800;">Linear Regression</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown("""
        <div class="kpi-card-clean">
            <div class="kpi-icon">🎯</div>
            <div class="kpi-label">Best MAE</div>
            <div class="kpi-value">8.49 <span style="font-size:1rem;">min</span></div>
            <div style="font-size: 0.78rem; color: #064E3B; font-weight: 800;">Mean Abs Error</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Workflow System
    st.markdown("""
    <div class="content-box">
        <div class="content-header">⚙️ System Architecture Pipeline</div>
        <div class="wf-grid">
            <div class="wf-node-item">
                <span class="wf-step-num">STEP 01</span>
                <div class="wf-node-title">📋 Input Data</div>
                <div style="font-size: 0.8rem; color: #D1D5DB; margin-top: 4px;">User enters parameters</div>
            </div>
            <div class="wf-node-item">
                <span class="wf-step-num">STEP 02</span>
                <div class="wf-node-title">🔄 Preprocessing</div>
                <div style="font-size: 0.8rem; color: #D1D5DB; margin-top: 4px;">Median Impute + One-Hot</div>
            </div>
            <div class="wf-node-item">
                <span class="wf-step-num">STEP 03</span>
                <div class="wf-node-title">🤖 ML Pipeline</div>
                <div style="font-size: 0.8rem; color: #D1D5DB; margin-top: 4px;">Linear Regression Fit</div>
            </div>
            <div class="wf-node-item">
                <span class="wf-step-num">STEP 04</span>
                <div class="wf-node-title">🎯 Prediction</div>
                <div style="font-size: 0.8rem; color: #D1D5DB; margin-top: 4px;">Estimated delivery duration</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. DELIVERY TIME PREDICTION PAGE
# ---------------------------------------------------------
elif page == "🎯 Delivery Time Prediction":
    st.markdown("""
    <div class="header-hero-card">
        <div class="hero-title">Delivery Time Prediction</div>
        <div class="hero-subtitle">Input order parameters to compute instant ML duration estimates</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    with st.form("prediction_form"):
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.markdown('<div style="font-family:\'Outfit\'; font-weight:800; font-size:1.25rem; color:#064E3B; margin-bottom:1rem;">🛵 Courier & Delivery Factors</div>', unsafe_allow_html=True)
            
            age = st.number_input(
                "Delivery Person Age",
                min_value=FEATURE_RANGES["Delivery_Person_Age"]["min"],
                max_value=FEATURE_RANGES["Delivery_Person_Age"]["max"],
                value=FEATURE_RANGES["Delivery_Person_Age"]["default"],
                help="Courier age (Standard operational range: 18-70)"
            )
            
            rating = st.number_input(
                "Delivery Person Rating (1.0 - 5.0)",
                min_value=FEATURE_RANGES["Delivery_Person_Rating"]["min"],
                max_value=FEATURE_RANGES["Delivery_Person_Rating"]["max"],
                value=FEATURE_RANGES["Delivery_Person_Rating"]["default"],
                step=FEATURE_RANGES["Delivery_Person_Rating"]["step"]
            )
            
            multiple_deliveries = st.number_input(
                "Simultaneous Deliveries Count",
                min_value=FEATURE_RANGES["Multiple_Deliveries"]["min"],
                max_value=FEATURE_RANGES["Multiple_Deliveries"]["max"],
                value=FEATURE_RANGES["Multiple_Deliveries"]["default"]
            )
            
            distance = st.number_input(
                "Delivery Distance (Kilometers)",
                min_value=FEATURE_RANGES["Distance_km"]["min"],
                max_value=FEATURE_RANGES["Distance_km"]["max"],
                value=FEATURE_RANGES["Distance_km"]["default"],
                step=FEATURE_RANGES["Distance_km"]["step"]
            )

        with col_b:
            st.markdown('<div style="font-family:\'Outfit\'; font-weight:800; font-size:1.25rem; color:#0A0A0A; margin-bottom:1rem;">🌧️ Environment & Order Parameters</div>', unsafe_allow_html=True)
            
            weather = st.selectbox("Weather Condition", options=FEATURE_RANGES["Weather_options"], index=0)
            traffic = st.selectbox("Traffic Congestion Level", options=FEATURE_RANGES["Traffic_options"], index=1)
            vehicle = st.selectbox("Vehicle Type", options=FEATURE_RANGES["Vehicle_options"], index=0)
            area = st.selectbox("Area Designation", options=FEATURE_RANGES["Area_options"], index=0)
            order_type = st.selectbox("Order Category", options=FEATURE_RANGES["Order_options"], index=1)
            festival = st.selectbox("Festival Period", options=FEATURE_RANGES["Festival_options"], index=0)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🚀 Predict Delivery Duration")

    st.markdown('</div>', unsafe_allow_html=True)

    # Validation Warnings
    if age < 18 or age > 70:
        st.warning("⚠️ Courier age is outside standard operational bounds (18-70).")
    if rating < 1.0 or rating > 5.0:
        st.warning("⚠️ Rating must be between 1.0 and 5.0.")
    if distance <= 0:
        st.warning("⚠️ Distance must be greater than 0.")

    if submit_btn:
        input_data = {
            "Delivery_Person_Age": float(age),
            "Delivery_Person_Rating": float(rating),
            "Weather": weather,
            "Traffic_Level": traffic,
            "Vehicle_Type": vehicle,
            "Area_Type": area,
            "Order_Type": order_type,
            "Festival": festival,
            "Multiple_Deliveries": float(multiple_deliveries),
            "Distance_km": float(distance)
        }

        try:
            pred_minutes = predict_delivery_time(model, input_data)
            rounded_mins = int(round(pred_minutes))
            formatted_dur = format_duration(pred_minutes)
            status_label, status_color = get_interpretation_badge(pred_minutes)

            # Prediction Outcome Box
            st.markdown(f"""
            <div class="prediction-green-hero">
                <div class="pred-hero-label">Estimated Delivery Duration</div>
                <div class="pred-hero-number">{rounded_mins} <span style="font-size: 2.2rem;">MINUTES</span></div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin-bottom: 1.2rem;">{formatted_dur} <span style="font-size:0.95rem; opacity:0.9;">({pred_minutes:.2f} exact)</span></div>
                <div style="display:inline-block; padding:8px 26px; border-radius:9999px; background-color:#0A0A0A; color:#FFFFFF; border:2px solid #FFFFFF; font-weight:800; font-size:1.05rem;">
                    {status_label}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.caption("ℹ️ **Demonstration Status Ranges:** 0–30 min: Fast delivery | 31–60 min: Moderate delivery time | 61–90 min: Longer delivery time | > 90 min: High delivery time.")

            # Input Summary Table
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.markdown('<div class="content-header">📋 Input Parameter Summary</div>', unsafe_allow_html=True)
            
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                st.markdown(f"• <strong style='color:#064E3B;'>Age:</strong> {age} years", unsafe_allow_html=True)
                st.markdown(f"• <strong style='color:#064E3B;'>Rating:</strong> {rating} / 5.0", unsafe_allow_html=True)
                st.markdown(f"• <strong style='color:#064E3B;'>Active Deliveries:</strong> {multiple_deliveries}", unsafe_allow_html=True)
                st.markdown(f"• <strong style='color:#064E3B;'>Distance:</strong> {distance} km", unsafe_allow_html=True)
                st.markdown(f"• <strong style='color:#064E3B;'>Vehicle:</strong> {vehicle}", unsafe_allow_html=True)
            with s_col2:
                st.markdown(f"• <strong style='color:#0A0A0A;'>Weather:</strong> {weather}", unsafe_allow_html=True)
                st.markdown(f"• <strong style='color:#0A0A0A;'>Traffic Level:</strong> {traffic}", unsafe_allow_html=True)
                st.markdown(f"• <strong style='color:#0A0A0A;'>Area:</strong> {area}", unsafe_allow_html=True)
                st.markdown(f"• <strong style='color:#0A0A0A;'>Order Type:</strong> {order_type}", unsafe_allow_html=True)
                st.markdown(f"• <strong style='color:#0A0A0A;'>Festival:</strong> {festival}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Export Button
            csv_bytes = generate_csv_report(input_data, pred_minutes)
            st.download_button(
                label="📥 Export Summary Report (CSV)",
                data=csv_bytes,
                file_name=f"delivery_prediction_{int(datetime.datetime.now().timestamp())}.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Prediction processing error: {str(e)}")

# ---------------------------------------------------------
# 3. MODEL PERFORMANCE PAGE
# ---------------------------------------------------------
elif page == "📊 Model Performance":
    st.markdown("""
    <div class="header-hero-card">
        <div class="hero-title">Model Evaluation Dashboard</div>
        <div class="hero-subtitle">Empirical benchmark comparison across evaluated regression models</div>
    </div>
    """, unsafe_allow_html=True)

    results_df = pd.DataFrame(MODEL_RESULTS).T.reset_index()
    results_df.columns = ["Model Algorithm", "MAE (min)", "MSE", "RMSE (min)", "R² Score", "Selected"]

    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown('<div class="content-header">📋 Benchmark Metric Comparison Table</div>', unsafe_allow_html=True)
    st.dataframe(results_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Combined Metric Plotly Figure (Dark Green & Black Palette)
    st.subheader("📈 Multi-Metric Benchmark Comparison Dashboard")
    
    models = list(MODEL_RESULTS.keys())
    maes = [MODEL_RESULTS[m]["MAE"] for m in models]
    rmses = [MODEL_RESULTS[m]["RMSE"] for m in models]
    r2s = [MODEL_RESULTS[m]["R2"] for m in models]

    model_colors = ['#064E3B' if MODEL_RESULTS[m]["Selected"] else '#121212' for m in models]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("MAE (Lower is Better)", "RMSE (Lower is Better)", "R² Score (Higher is Better)"),
        horizontal_spacing=0.08
    )

    fig.add_trace(go.Bar(x=models, y=maes, marker_color=model_colors, name="MAE (min)", text=maes, textposition='auto'), row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=rmses, marker_color=model_colors, name="RMSE (min)", text=rmses, textposition='auto'), row=1, col=2)
    fig.add_trace(go.Bar(x=models, y=r2s, marker_color=model_colors, name="R² Score", text=r2s, textposition='auto'), row=1, col=3)

    fig.update_layout(
        height=480,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=110),
        **PLOTLY_GREEN_THEME
    )
    fig.update_xaxes(tickangle=-25)
    st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    st.caption("🟩 **Dark Green Bar:** Highlights the selected production model (**Linear Regression**).")

    # Actual vs Predicted Plot
    st.subheader("🎯 Actual vs. Predicted Delivery Time")
    df_test_preds = load_test_predictions()

    if not df_test_preds.empty and "Actual_Delivery_Time" in df_test_preds.columns and "Predicted_Delivery_Time" in df_test_preds.columns:
        fig_scatter = px.scatter(
            df_test_preds,
            x="Actual_Delivery_Time",
            y="Predicted_Delivery_Time",
            opacity=0.75,
            color_discrete_sequence=['#064E3B'],
            labels={"Actual_Delivery_Time": "Actual Delivery Duration (minutes)", "Predicted_Delivery_Time": "Predicted Delivery Duration (minutes)"},
            title="Linear Regression: Actual vs Predicted Delivery Duration"
        )
        
        min_val = min(df_test_preds["Actual_Delivery_Time"].min(), df_test_preds["Predicted_Delivery_Time"].min())
        max_val = max(df_test_preds["Actual_Delivery_Time"].max(), df_test_preds["Predicted_Delivery_Time"].max())
        
        fig_scatter.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Ideal Reference (y = x)",
                line=dict(color="#0A0A0A", dash="dash", width=2)
            )
        )
        fig_scatter.update_layout(height=480, **PLOTLY_GREEN_THEME)
        st.plotly_chart(fig_scatter, use_container_width=True, config={'responsive': True})
        st.info("💡 **Graph Analysis:** Points hugging the dashed black line ($y = x$) demonstrate accurate model predictions across delivery durations.")
    else:
        st.warning("Test prediction dataset is not available.")

# ---------------------------------------------------------
# 4. FEATURE ANALYSIS PAGE
# ---------------------------------------------------------
elif page == "🔍 Feature Analysis":
    st.markdown("""
    <div class="header-hero-card">
        <div class="hero-title">Feature Importance & Coefficients</div>
        <div class="hero-subtitle">Model coefficient weights extracted dynamically from the fitted Linear Regression pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    df_coef = extract_coefficients(model)

    if not df_coef.empty:
        top_10 = df_coef.head(10).sort_values(by="Coefficient", ascending=True)
        color_list = ['#064E3B' if c > 0 else '#0A0A0A' for c in top_10["Coefficient"]]

        fig_bar = go.Figure(
            go.Bar(
                x=top_10["Coefficient"],
                y=top_10["Feature"],
                orientation='h',
                marker_color=color_list,
                text=[f"{c:+.2f} min" for c in top_10["Coefficient"]],
                textposition='outside'
            )
        )
        fig_bar.update_layout(
            title="Top 10 Feature Impact Coefficients (Linear Regression)",
            xaxis_title="Fitted Coefficient Value (Estimated Duration Impact in Minutes)",
            yaxis_title="Feature Name",
            height=500,
            **PLOTLY_GREEN_THEME
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'responsive': True})

        st.markdown("""
        <div class="content-box">
            <div class="content-header">💡 Coefficient Interpretation Guidelines</div>
            <ul style="color: #1F2937; font-size: 1rem; line-height: 1.8; font-weight: 500;">
                <li><strong style="color:#064E3B;">Positive Coefficients (Green Bar):</strong> Associated with an increase in predicted delivery duration relative to baseline (e.g. <code>Traffic Level Jam (+15.78 min)</code>, <code>Stormy Weather (+8.88 min)</code>).</li>
                <li><strong style="color:#0A0A0A;">Negative Coefficients (Black Bar):</strong> Associated with a reduction in predicted delivery duration relative to baseline (e.g. <code>Low Traffic (-13.26 min)</code>, <code>Sunny Weather (-6.63 min)</code>).</li>
                <li><strong>Magnitude:</strong> Larger absolute values reflect greater model weighting within the fitted linear equation.</li>
            </ul>
            <p style="font-size:0.85rem; color:#6B7280; margin-top:0.8rem;">
                ⚠️ <em>Statistical Disclaimer: Coefficients indicate model relationships and should not be interpreted as strict causal effects.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Unable to extract coefficients from model.")

# ---------------------------------------------------------
# 5. DATASET & METHODOLOGY PAGE
# ---------------------------------------------------------
elif page == "📋 Dataset & Methodology":
    st.markdown("""
    <div class="header-hero-card">
        <div class="hero-title">Dataset & ML Methodology</div>
        <div class="hero-subtitle">Data processing pipeline architecture and model evaluation summary</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-box">
        <div class="content-header">🔄 Preprocessing & Engineering Pipeline</div>
        <p style="color:#1F2937; font-weight:500;">
            To ensure complete reproducibility and avoid data leakage, data cleaning and feature transformations 
            are encapsulated directly inside a Scikit-Learn <code>Pipeline</code> containing a <code>ColumnTransformer</code>.
        </p>
        <div style="background:#0A0A0A; border:2px solid #064E3B; padding:1.2rem; border-radius:12px; font-family:monospace; font-size:0.92rem; color:#FFFFFF;">
            Raw Input Data (10 features)<br>
            ├── Numerical Pipeline ➔ SimpleImputer(strategy='median')<br>
            └── Categorical Pipeline ➔ SimpleImputer(strategy='most_frequent') ➔ OneHotEncoder(handle_unknown='ignore')<br>
            ↓<br>
            Fitted LinearRegression() Model ➔ Prediction Output (Delivery_Time_min)
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-box">
        <div class="content-header">🔑 Verified Project Findings</div>
        <ol style="color:#1F2937; font-size:1rem; line-height:1.8; font-weight:500;">
            <li><strong>Linear Regression</strong> achieved the best overall performance with an R² of <strong>0.65</strong>.</li>
            <li><strong>Linear Regression</strong> achieved the lowest Mean Absolute Error of <strong>8.49 minutes</strong>.</li>
            <li><strong>Gradient Boosting</strong> performed closely with an R² of <strong>0.64</strong>.</li>
            <li><strong>Random Forest</strong> achieved an R² score of <strong>0.61</strong>.</li>
            <li><strong>Decision Tree</strong> underperformed with an R² score of <strong>0.34</strong>.</li>
            <li><strong>Traffic Congestion</strong> (Jam level) demonstrated the strongest positive weight (+15.78 minutes).</li>
            <li><strong>Distance (km)</strong> exhibited a consistent positive correlation (+2.59 minutes per km).</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 6. ABOUT THE PROJECT PAGE
# ---------------------------------------------------------
elif page == "👨‍💻 About the Project":
    st.markdown("""
    <div class="header-hero-card">
        <div class="hero-title">About This Project</div>
        <div class="hero-subtitle">Executive project details, tech stack overview, developer GitHub, and model metadata</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("""
        <div class="content-box">
            <div class="content-header">🎯 Project Objective</div>
            <p style="color:#1F2937; line-height:1.7; font-weight:500;">
                To design, evaluate, and deploy an end-to-end Machine Learning regression solution capable of 
                predicting food delivery duration based on courier metrics, environmental variables, traffic conditions, 
                order attributes, and geographical distance.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="content-box">
            <div class="content-header">🛠️ Technology Stack</div>
            <p style="color:#1F2937; line-height:1.7; font-weight:500;">
                • <strong>Core:</strong> Python 3.12, Streamlit<br>
                • <strong>Data & ML:</strong> Pandas, NumPy, Scikit-Learn 1.6.1, Joblib<br>
                • <strong>Data Viz:</strong> Plotly Express, Plotly Subplots
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-box">
        <div class="content-header">👨‍💻 Developer & GitHub Account</div>
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
            <div>
                <div style="font-size:1.15rem; font-weight:800; color:#0A0A0A;">GitHub Profile: Abrar-Codes404</div>
                <div style="color:#6B7280; font-weight:500; font-size:0.95rem;">Repository Owner & Project Maintainer</div>
            </div>
            <a href="https://github.com/Abrar-Codes404" target="_blank" style="text-decoration:none;">
                <div style="background:#064E3B; color:#FFFFFF; padding:10px 22px; border-radius:10px; font-weight:800; display:flex; align-items:center; gap:8px;">
                    <span>🐙 View GitHub Profile (@Abrar-Codes404)</span>
                </div>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="content-box">
        <div class="content-header">🏥 Model Technical Specifications</div>
        <div style="background:#FAFAFA; border-left:5px solid #064E3B; border:2px solid #0A0A0A; padding:1.2rem; border-radius:8px; font-size:0.95rem; color:#0A0A0A; font-weight:500;">
            • <strong>Model File Name:</strong> <code>food_delivery_linear_regression.pkl</code><br>
            • <strong>Pipeline Architecture:</strong> <code>sklearn.pipeline.Pipeline</code><br>
            • <strong>Scikit-Learn Version:</strong> <code>1.6.1</code><br>
            • <strong>Transformers:</strong> <code>SimpleImputer(median)</code>, <code>SimpleImputer(most_frequent)</code>, <code>OneHotEncoder(handle_unknown='ignore')</code><br>
            • <strong>Final Regressor:</strong> <code>LinearRegression()</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.88rem; padding-top: 2rem; font-weight: 600;">
    Food Delivery Time Prediction System • Developed by <a href="https://github.com/Abrar-Codes404" target="_blank" style="color:#064E3B; text-decoration:none; font-weight:800;">Abrar (@Abrar-Codes404)</a> • Built with Streamlit, Plotly & Scikit-Learn
</div>
""", unsafe_allow_html=True)
