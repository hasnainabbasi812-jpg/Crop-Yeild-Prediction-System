import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page Configuration & Dark Theme Layout
st.set_page_config(page_title="AgriSmart - Crop Yield Prediction", page_icon="🌾", layout="wide")

# Custom CSS for Premium Black & Dark Grey Look
st.markdown("""
    <style>
    .main { background-color: #0A0A0A; color: #E0E0E0; }
    .stApp { background-color: #0A0A0A; }
    h1, h2, h3 { color: #FFFFFF !important; font-family: 'Segoe UI', sans-serif; }
    div.stButton > button:first-child {
        background-color: #1E1E1E; color: #00FF66; border: 1px solid #333333; border-radius: 6px;
        font-weight: bold; width: 100%; padding: 10px;
    }
    div.stButton > button:first-child:hover { background-color: #2D2D2D; border-color: #00FF66; color: #FFFFFF; }
    div[data-testid="stSidebar"] { background-color: #121212; border-right: 1px solid #222222; }
    .stMetric { background-color: #161616; padding: 15px; border-radius: 8px; border: 1px solid #222222; }
    </style>
""", unsafe_allow_html=True)

# Load Saved Model and Scaler Safely
@st.cache_resource
def load_assets():
    model_path = 'crop_gb_model.pkl'
    scaler_path = 'scaler.pkl'
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    return None, None

model, scaler = load_assets()

# Sidebar Navigation Panel (7 Pages)
st.sidebar.markdown("<h2 style='text-align: center; color: #00FF66;'>🌱 AgriSmart Menu</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to Page:", [
    "1. Home Dashboard", 
    "2. Yield Predictor Engine", 
    "3. Model Performance", 
    "4. Explanatory Data Analysis", 
    "5. Feature Engineering Metrics", 
    "6. Project Overview",
    "7. Student & Submission Info"
])

# ----------------- PAGE 1: HOME DASHBOARD -----------------
if page == "1. Home Dashboard":
    st.title("🌾 Crop Yield Prediction System")
    st.write("AIC-221 Introduction to Machine Learning — Final Presentation Dashboard")
    st.markdown("---")
    st.markdown("### Welcome to the Interactive Precision Agriculture System")
    st.write("Is web application ka maqsad predictive analytics aur advanced machine learning boosting techniques ki madad se crop yield prediction ko asan aur accurate banana hai.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Model Optimization (R² Score)", value="88.96%", delta="Gradient Boosting")
    with col2:
        st.metric(label="System Mean Error (MAPE)", value="8.67%", delta="Highly Accurate")
    with col3:
        st.metric(label="Dataset Rows Supported", value="1,200 Records", delta="Kaggle Verified")

# ----------------- PAGE 2: YIELD PREDICTOR ENGINE -----------------
elif page == "2. Yield Predictor Engine":
    st.title("🔮 Interactive Yield Prediction Engine")
    st.write("Niche diye gaye inputs field fill karein taake saved machine learning model crop production ka estimate calculate kar sake.")
    st.markdown("---")
    
    if model is None:
        st.error("⚠️ System Alert: `crop_gb_model.pkl` ya `scaler.pkl` files is directory mein majood nahi hain. Kindly unhein upload karein.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            area = st.number_input("Cultivated Area (Hectares)", min_value=10.0, max_value=10000.0, value=1250.0, step=10.0)
            rainfall = st.number_input("Annual Rainfall (mm)", min_value=100.0, max_value=3000.0, value=1150.0, step=50.0)
            fertilizer = st.number_input("Fertilizer Intensive Usage (kg/ha)", min_value=10.0, max_value=1000.0, value=280.0, step=5.0)
            pesticide = st.number_input("Pesticide Applied Ratio (kg/ha)", min_value=0.1, max_value=20.0, value=4.5, step=0.1)
        
        with col2:
            crop_year = st.slider("Cultivation Year", 2010, 2026, 2026)
            crop = st.selectbox("Select Target Crop Type", ['Rice','Wheat','Maize','Sugarcane','Cotton','Barley','Mustard','Groundnut'])
            season = st.selectbox("Select Agricultural Season", ['Kharif','Rabi','Whole Year','Summer','Winter','Autumn'])
            state = st.selectbox("Select Regional State", ['Punjab','Haryana','UP','Rajasthan','Maharashtra','Gujarat','MP','AP'])

        crop_list = ['Rice','Wheat','Maize','Sugarcane','Cotton','Barley','Mustard','Groundnut']
        season_list = ['Kharif','Rabi','Whole Year','Summer','Winter','Autumn']
        state_list = ['Punjab','Haryana','UP','Rajasthan','Maharashtra','Gujarat','MP','AP']
        
        crop_enc = crop_list.index(crop)
        season_enc = season_list.index(season)
        state_enc = state_list.index(state)
        
        rainfall_per_area = rainfall / area
        fertilizer_per_area = fertilizer / area
        soil_quality = (fertilizer * 0.6 + rainfall * 0.4) / 100
        
        input_df = pd.DataFrame([[area, rainfall, fertilizer, pesticide, crop_year, 
                                  crop_enc, season_enc, state_enc, 
                                  rainfall_per_area, fertilizer_per_area, soil_quality]])
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Run Analytical Prediction Model"):
            prediction = model.predict(input_df)[0]
            st.balloons()
            st.markdown(f"""
                <div style='background-color: #111111; padding: 20px; border-radius: 10px; border: 1px solid #00FF66; text-align: center;'>
                    <h2 style='color: #FFFFFF; margin: 0;'>🎉 Estimated Crop Production Output</h2>
                    <h1 style='color: #00FF66; margin: 10px 0;'>{prediction:.2f} Tonnes</h1>
                    <p style='color: #888888; margin: 0;'>Gradient Boosting Regressor Inference Engine</p>
                </div>
            """, unsafe_allow_html=True)

# ----------------- PAGE 3: MODEL PERFORMANCE -----------------
elif page == "3. Model Performance":
    st.title("🤖 Evaluated Models & Architecture Comparison")
    st.write("AIC-221 course guidelines ke mutabiq test kiye gaye tamam algorithms ke standard evaluative metrics:")
    st.markdown("---")
    
    performance_matrix = {
        'Algorithm Evaluated': ['Linear Regression', 'Ridge Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting (Best)', 'Support Vector Regressor (SVR)'],
        'MAE (Lower is Better)': [78.36, 78.39, 87.13, 72.66, 66.89, 69.28],
        'RMSE (Error Variance)': [101.40, 101.31, 111.96, 95.40, 89.07, 90.76],
        'R² Accuracy Score': [0.8569, 0.8572, 0.8256, 0.8734, 0.8896, 0.8854],
        'MAPE % Error': ["11.03%", "11.04%", "11.58%", "9.76%", "8.67%", "10.08%"],
        '5-Fold Cross Validation R²': [0.8593, 0.8593, 0.7924, 0.8678, 0.8780, 0.8516]
    }
    st.dataframe(pd.DataFrame(performance_matrix), use_container_width=True)

# ----------------- PAGE 4: EXPLANATORY DATA ANALYSIS -----------------
elif page == "4. Explanatory Data Analysis":
    st.title("📈 Visual Data Analysis Insights")
    st.markdown("---")
    if os.path.exists('crop_yield_analysis.png'):
        st.image('crop_yield_analysis.png', caption='Final Complete Model Evaluation & Validation Dashboards', use_container_width=True)
    else:
        st.warning("📊 Information: `crop_yield_analysis.png` dashboard file project main directory mein copy nahi mili.")

# ----------------- PAGE 5: FEATURE ENGINEERING METRICS -----------------
elif page == "5. Feature Engineering Metrics":
    st.title("🧠 Feature Engineering & Mathematical Interventions")
    st.markdown("---")
    st.markdown("#### 1. Soil Quality Index (SQI)")
    st.code("Soil_Quality_Index = (Fertilizer * 0.6 + Annual_Rainfall * 0.4) / 100", language="python")
    st.markdown("#### 2. Rainfall per Unit Area")
    st.code("Rainfall_per_Area = Annual_Rainfall / Area", language="python")
    st.markdown("#### 3. Fertilizer Intensity Ratio")
    st.code("Fertilizer_per_Area = Fertilizer / Area", language="python")

# ----------------- PAGE 6: PROJECT OVERVIEW -----------------
elif page == "6. Project Overview":
    st.title("ℹ️ Machine Learning Pipeline Architecture")
    st.markdown("---")
    st.write("1. Data Cleaning (Median Imputation)\n2. Encoding Vectors (Label Encoding)\n3. Scaling (StandardScaler)\n4. Model Selection & Cross Validation")

# ----------------- PAGE 7: STUDENT & SUBMISSION INFO -----------------
elif page == "7. Contact & Submission Info":
    st.title("🎓 Project Authentication & Metadata")
    st.markdown("---")
    st.markdown("### 👤 Student Information")
    st.write("**Name:** Hasnain")
    st.write("**Course ID:** AIC-221 Intro to Machine Learning")
    st.write("**Batch Details:** AI-SP-24 | Semester: 6th")
    st.write("**Supervisor:** Instructor Abdul Baqi Malik")
