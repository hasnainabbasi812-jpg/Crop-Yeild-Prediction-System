import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page Configuration
st.set_page_config(page_title="AgriSmart - Crop Yield Prediction", page_icon="🌾", layout="wide")

# CSS for Dark Theme
st.markdown("""
    <style>
    .main { background-color: #0A0A0A; color: #E0E0E0; }
    .stApp { background-color: #0A0A0A; }
    h1, h2, h3 { color: #FFFFFF !important; font-family: 'Segoe UI', sans-serif; }
    div.stButton > button:first-child { background-color: #1E1E1E; color: #00FF66; border: 1px solid #333333; border-radius: 6px; font-weight: bold; width: 100%; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

# Load Model using Joblib
@st.cache_resource
def load_assets():
    model_path = 'crop_gb_model_final.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_assets()

# Sidebar Navigation
page = st.sidebar.radio("Go to Page:", ["1. Home Dashboard", "2. Yield Predictor Engine"])

# Page 1
if page == "1. Home Dashboard":
    st.title("🌾 Crop Yield Prediction System")
    st.write("Welcome to AgriSmart! Use the sidebar to navigate.")

# Page 2
elif page == "2. Yield Predictor Engine":
    st.title("🔮 Yield Predictor")
    if model is None:
        st.error("⚠️ Model file `crop_gb_model_final.joblib` not found in directory!")
    else:
        area = st.number_input("Area (Hectares)", value=1250.0)
        rainfall = st.number_input("Rainfall (mm)", value=1150.0)
        fertilizer = st.number_input("Fertilizer (kg/ha)", value=280.0)
        pesticide = st.number_input("Pesticide (kg/ha)", value=4.5)
        crop_year = st.slider("Year", 2010, 2026, 2026)
        
        # Input Data Preparation
        input_data = np.array([[area, rainfall, fertilizer, pesticide, crop_year, 0, 0, 0, (rainfall/area), (fertilizer/area), 0.5]])
        
        if st.button("🚀 Run Analytical Prediction Model"):
            prediction = model.predict(input_data)[0]
            st.success(f"Estimated Output: {prediction:.2f} Tonnes")
