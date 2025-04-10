import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Set layout
st.set_page_config(layout="wide")

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/amaggiacomo11/capstone/main/Capstone_flu%2804092025%29.csv"
    return pd.read_csv(url)

df = load_data()

st.title("Flu Outbreak Risk Predictor")
st.write("Use this app to estimate your area's flu risk and get recommended safety actions based on public data and model predictions.")

# --------------------------------------
# Model Setup
# --------------------------------------

# Target and features
X = df.drop(columns=[
    "cases_per_100k", "estimated_positive_cases", "Percent_Positive",
    "Total_population", "Total_Specimens", "Week", "LandArea_SqMi",
    "Month", "State"
])
y = df["cases_per_100k"]

# Choose some key features to keep things simple
selected_features = ["AVG_TEMP", "Pop_Density", "Week_sin", "Week_cos"]

X = X[selected_features]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), selected_features)
])

# Final pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42))
])

# Train model
model.fit(X_train, y_train)

# --------------------------------------
# User Input
# --------------------------------------

st.sidebar.header("Your Information")

state = st.sidebar.selectbox("Select your state", sorted(df["State"].unique()))
week = st.sidebar.slider("Week of the year", 1, 52, 10)
avg_temp = st.sidebar.slider("Average weekly temperature (°F)", -20.0, 120.0, 60.0)
pop_density = st.sidebar.slider("Population density (people per sq mi)", float(df["Pop_Density"].min()), float(df["Pop_Density"].max()), float(df["Pop_Density"].mean()))

# --------------------------------------
# Prediction
# --------------------------------------

# Feature engineering: convert week to sin/cos
import math
week_sin = math.sin(2 * math.pi * week / 52)
week_cos = math.cos(2 * math.pi * week / 52)

# Format input for model
user_input = pd.DataFrame({
    "AVG_TEMP": [avg_temp],
    "Pop_Density": [pop_density],
    "Week_sin": [week_sin],
    "Week_cos": [week_cos]
})

predicted_cases = model.predict(user_input)[0]

# Risk classification
def get_risk_level(cases):
    if cases < 10:
        return "Minimal", "🟢"
    elif cases < 25:
        return "Low", "🟡"
    elif cases < 50:
        return "Moderate", "🟠"
    else:
        return "High", "🔴"

risk_level, risk_emoji = get_risk_level(predicted_cases)

# --------------------------------------
# Mitigation Advice
# --------------------------------------

mitigation = {
    "Minimal": "🟢 Continue healthy habits and stay updated on flu activity.",
    "Low": "🟡 Wash hands frequently and consider avoiding large crowds during peak times.",
    "Moderate": "🟠 Limit indoor gatherings, wear a mask in public areas, and consider vaccination if not already done.",
    "High": "🔴 Avoid crowded spaces, mask up indoors, practice strict hygiene, and consider remote work/school if possible."
}

# --------------------------------------
# Display Results
# --------------------------------------

st.subheader("Prediction Results")
st.markdown(f"**Predicted flu cases per 100k:** `{predicted_cases:.2f}`")
st.markdown(f"## Risk level: {risk_emoji} **{risk_level}**")

st.divider()

st.subheader("Recommended Mitigation Strategy")
st.write(mitigation[risk_level])
