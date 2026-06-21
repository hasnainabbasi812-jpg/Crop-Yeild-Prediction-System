import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="AgriSmart - Crop Yield Prediction", page_icon="🌾", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0A0A0A; color: #E0E0E0; }
    .stApp { background-color: #0A0A0A; }
    h1, h2, h3 { color: #FFFFFF !important; font-family: 'Segoe UI', sans-serif; }
    div.stButton > button:first-child { background-color: #1E1E1E; color: #00FF66; border: 1px solid #333333; border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model_path = 'crop_gb_model_final.joblib'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    return None

model = load_assets()

st.sidebar.markdown("<h2 style='text-align: center; color: #00FF66;'>🌱 AgriSmart Menu</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to Page:", [
    "1. Home Dashboard",
    "2. Yield Predictor Engine",
    "3. Model Performance",
    "4. Explanatory Data Analysis",
    "5. Feature Engineering Metrics",
    "6. Project Overview",
    "7. Contact & Submission Info"
])

if page == "1. Home Dashboard":
    st.title("🌾 Crop Yield Prediction System")
    st.write("AIC-221 Final Presentation Dashboard")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Optimization", "88.96%", "Gradient Boosting")
    col2.metric("System MAPE", "8.67%", "Highly Accurate")
    col3.metric("Dataset Rows", "1,200", "Kaggle Verified")

elif page == "2. Yield Predictor Engine":
    st.title("🔮 Interactive Yield Prediction Engine")
    if model is None:
        st.error("System assets not found.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            area = st.number_input("Area (Hectares)", 10.0, 10000.0, 1250.0)
            rainfall = st.number_input("Rainfall (mm)", 100.0, 3000.0, 1150.0)
            fertilizer = st.number_input("Fertilizer (kg/ha)", 10.0, 1000.0, 280.0)
            pesticide = st.number_input("Pesticide (kg/ha)", 0.1, 20.0, 4.5)
        with col2:
            crop_year = st.slider("Year", 2010, 2026, 2026)
            crop = st.selectbox("Crop", ['Rice','Wheat','Maize','Sugarcane','Cotton','Barley','Mustard','Groundnut'])
            season = st.selectbox("Season", ['Kharif','Rabi','Whole Year','Summer','Winter','Autumn'])
            state = st.selectbox("State", ['Punjab','Haryana','UP','Rajasthan','Maharashtra','Gujarat','MP','AP'])

        crop_enc = ['Rice','Wheat','Maize','Sugarcane','Cotton','Barley','Mustard','Groundnut'].index(crop)
        season_enc = ['Kharif','Rabi','Whole Year','Summer','Winter','Autumn'].index(season)
        state_enc = ['Punjab','Haryana','UP','Rajasthan','Maharashtra','Gujarat','MP','AP'].index(state)

        if st.button("🚀 Run Prediction"):
            input_data = np.array([[area, rainfall, fertilizer, pesticide, crop_year, crop_enc, season_enc, state_enc, rainfall/area, fertilizer/area, (fertilizer*0.6+rainfall*0.4)/100]])
            prediction = model.predict(input_data)[0]
            st.balloons()
            st.success(f"Estimated Production: {prediction:.2f} Tonnes")

elif page == "7. Contact & Submission Info":
    st.title("🎓 Project Authentication")
    st.write("**Name:** Hasnain")
    st.write("**Course:** AIC-221")
    st.write("**Submission Date:** June 11, 2026")
