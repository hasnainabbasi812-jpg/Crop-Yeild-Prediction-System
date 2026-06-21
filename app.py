import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="AgriSmart - Crop Yield Prediction", page_icon="🌾", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0A0A0A; color: #E0E0E0; }
    .stApp { background-color: #0A0A0A; }
    h1, h2, h3 { color: #FFFFFF !important; font-family: 'Segoe UI', sans-serif; }
    div.stButton > button:first-child { background-color: #1E1E1E; color: #00FF66; border: 1px solid #333333; border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

CROPS = ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton', 'Barley', 'Mustard', 'Groundnut']
SEASONS = ['Kharif', 'Rabi', 'Whole Year', 'Summer', 'Winter', 'Autumn']
STATES = ['Punjab', 'Haryana', 'UP', 'Rajasthan', 'Maharashtra', 'Gujarat', 'MP', 'AP']

# Realistic-ish base yield (tons/hectare) per crop, used only to generate training data
CROP_BASE_YIELD = {'Rice': 2.5, 'Wheat': 3.0, 'Maize': 2.8, 'Sugarcane': 60.0,
                    'Cotton': 1.5, 'Barley': 2.2, 'Mustard': 1.2, 'Groundnut': 1.8}
SEASON_FACTOR = {'Kharif': 1.0, 'Rabi': 1.05, 'Whole Year': 1.1, 'Summer': 0.9, 'Winter': 1.0, 'Autumn': 0.95}


# ----------------------------------------------------------------------
# Train model at startup (cached). No external .joblib file needed,
# so there is no scikit-learn version-mismatch risk.
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Training model on startup...")
def train_model():
    rng = np.random.default_rng(42)
    n = 4000

    area = rng.uniform(10, 10000, n)
    rainfall = rng.uniform(100, 3000, n)
    fertilizer = rng.uniform(10, 1000, n)
    pesticide = rng.uniform(0.1, 20, n)
    crop_year = rng.integers(2010, 2027, n)
    crop = rng.choice(CROPS, n)
    season = rng.choice(SEASONS, n)
    state = rng.choice(STATES, n)

    crop_enc = np.array([CROPS.index(c) for c in crop])
    season_enc = np.array([SEASONS.index(s) for s in season])
    state_enc = np.array([STATES.index(s) for s in state])

    crop_base = np.array([CROP_BASE_YIELD[c] for c in crop])
    season_fac = np.array([SEASON_FACTOR[s] for s in season])

    yield_per_ha = (
        crop_base
        * (0.3 + 0.0004 * rainfall)
        * (0.5 + 0.001 * fertilizer)
        * (0.9 + 0.01 * pesticide)
        * season_fac
        * rng.normal(1.0, 0.05, n)
    )
    production = (yield_per_ha * area).clip(min=1)

    X = np.column_stack([
        area, rainfall, fertilizer, pesticide, crop_year,
        crop_enc, season_enc, state_enc,
        rainfall / area, fertilizer / area, (fertilizer * 0.6 + rainfall * 0.4) / 100
    ])
    y = production

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "R2": float(r2_score(y_test, y_pred)),
    }

    feature_names = ["Area", "Rainfall", "Fertilizer", "Pesticide", "Year",
                      "Crop", "Season", "State", "Rainfall/Area", "Fertilizer/Area", "Blended Index"]
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    train_df = pd.DataFrame({
        "Area": area, "Rainfall": rainfall, "Fertilizer": fertilizer,
        "Pesticide": pesticide, "Crop": crop, "Season": season,
        "State": state, "Production": production,
    })

    return model, metrics, importances, train_df


model, metrics, importances, train_df = train_model()

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

# ----------------------------------------------------------------------
# 1. Home Dashboard
# ----------------------------------------------------------------------
if page == "1. Home Dashboard":
    st.title("🌾 Crop Yield Prediction System")
    st.write("AIC-221 Final Presentation Dashboard")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model R²", f"{metrics['R2']*100:.2f}%", "Gradient Boosting")
    col2.metric("Test RMSE", f"{metrics['RMSE']:.1f} tons", "Lower is better")
    col3.metric("Training Rows", f"{len(train_df):,}", "Synthetic, realistic")
    st.markdown("---")
    st.write(
        "This dashboard predicts total crop **production (tonnes)** from field area, weather, "
        "and input usage. Use the sidebar to explore the predictor, model performance, and "
        "data analysis pages."
    )

# ----------------------------------------------------------------------
# 2. Yield Predictor Engine
# ----------------------------------------------------------------------
elif page == "2. Yield Predictor Engine":
    st.title("🔮 Interactive Yield Prediction Engine")
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area (Hectares)", 10.0, 10000.0, 1250.0)
        rainfall = st.number_input("Rainfall (mm)", 100.0, 3000.0, 1150.0)
        fertilizer = st.number_input("Fertilizer (kg/ha)", 10.0, 1000.0, 280.0)
        pesticide = st.number_input("Pesticide (kg/ha)", 0.1, 20.0, 4.5)
    with col2:
        crop_year = st.slider("Year", 2010, 2026, 2026)
        crop = st.selectbox("Crop", CROPS)
        season = st.selectbox("Season", SEASONS)
        state = st.selectbox("State", STATES)

    crop_enc = CROPS.index(crop)
    season_enc = SEASONS.index(season)
    state_enc = STATES.index(state)

    if st.button("🚀 Run Prediction"):
        input_data = np.array([[area, rainfall, fertilizer, pesticide, crop_year, crop_enc, season_enc, state_enc,
                                 rainfall / area, fertilizer / area, (fertilizer * 0.6 + rainfall * 0.4) / 100]])
        prediction = model.predict(input_data)[0]
        st.balloons()
        st.success(f"Estimated Production: {prediction:.2f} Tonnes")
        st.caption(f"≈ {prediction/area:.2f} tonnes/hectare for the selected crop and conditions")

# ----------------------------------------------------------------------
# 3. Model Performance
# ----------------------------------------------------------------------
elif page == "3. Model Performance":
    st.title("📊 Model Performance")
    st.write("Gradient Boosting Regressor evaluated on a held-out 20% test split.")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{metrics['RMSE']:.2f}")
    c2.metric("MAE", f"{metrics['MAE']:.2f}")
    c3.metric("R²", f"{metrics['R2']:.3f}")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(["RMSE", "MAE"], [metrics["RMSE"], metrics["MAE"]], color=["#00CC66", "#00FF66"])
    ax.set_facecolor("#0A0A0A")
    fig.patch.set_facecolor("#0A0A0A")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    for b, v in zip(bars, [metrics["RMSE"], metrics["MAE"]]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}", ha="center", va="bottom", color="white")
    st.pyplot(fig)

# ----------------------------------------------------------------------
# 4. Explanatory Data Analysis
# ----------------------------------------------------------------------
elif page == "4. Explanatory Data Analysis":
    st.title("🔍 Explanatory Data Analysis")
    st.dataframe(train_df.head(20), use_container_width=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax in axes:
        ax.set_facecolor("#0A0A0A")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
    fig.patch.set_facecolor("#0A0A0A")

    axes[0].hist(train_df["Production"], bins=30, color="#00FF66")
    axes[0].set_title("Production Distribution", color="white")

    avg_by_crop = train_df.groupby("Crop")["Production"].mean().sort_values()
    axes[1].barh(avg_by_crop.index, avg_by_crop.values, color="#00CC66")
    axes[1].set_title("Avg Production by Crop", color="white")

    plt.tight_layout()
    st.pyplot(fig)

# ----------------------------------------------------------------------
# 5. Feature Engineering Metrics
# ----------------------------------------------------------------------
elif page == "5. Feature Engineering Metrics":
    st.title("🌟 Feature Importance")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(importances.index[::-1], importances.values[::-1], color="#00FF66")
    ax.set_facecolor("#0A0A0A")
    fig.patch.set_facecolor("#0A0A0A")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    st.pyplot(fig)
    st.caption("Engineered features (Rainfall/Area, Fertilizer/Area, Blended Index) capture interaction effects "
               "between inputs and field size.")

# ----------------------------------------------------------------------
# 6. Project Overview
# ----------------------------------------------------------------------
elif page == "6. Project Overview":
    st.title("📘 Project Overview")
    st.markdown("""
    **Objective:** Predict total crop production (tonnes) from field area, weather conditions
    (rainfall), and input usage (fertilizer, pesticide), across multiple crops, seasons, and states.

    **Pipeline:**
    1. Feature engineering — raw inputs plus derived ratios (rainfall/area, fertilizer/area, blended index)
    2. Train/test split (80/20)
    3. Model training — Gradient Boosting Regressor
    4. Evaluation — RMSE, MAE, R²
    5. Deployment — this interactive Streamlit app

    **Why Gradient Boosting?** It handles non-linear relationships between weather/input variables
    and yield well, and typically outperforms a single decision tree or plain linear regression on
    tabular agricultural data.
    """)

# ----------------------------------------------------------------------
# 7. Contact & Submission Info
# ----------------------------------------------------------------------
elif page == "7. Contact & Submission Info":
    st.title("🎓 Project Authentication")
    st.write("**Name:** Hasnain")
    st.write("**Course:** AIC-221")
    st.write("**Submission Date:** June 11, 2026")
