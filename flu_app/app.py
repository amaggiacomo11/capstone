import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load dataset from GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/cfernandez3/CAPSTONE-DELOITTE-Project/refs/heads/main/Capstone_flu(04092025).csv"
    return pd.read_csv(url)

df = load_data()

st.set_page_config(page_title="Flu Outbreak Predictor", layout="wide")
st.title("🦠 Flu Outbreak Risk Predictor 🦠")
st.write("🌡️Like a weather app but for your flu exposure!🌡️")

# Define race and age group columns
race_columns = [
    "White",
    "Black or African American",
    "American Indian and Alaska Native",
    "Asian",
    "Native Hawaiian and Other Pacific Islander",
    "Some other race"
]

age_columns = [
    "Under 5 years", "5 to 9 years", "10 to 14 years", "15 to 19 years",
    "20 to 24 years", "25 to 34 years", "35 to 44 years", "45 to 54 years",
    "55 to 59 years", "60 to 64 years", "65 to 74 years", "75 to 84 years", "85 years and over"
]

# Sidebar – Personal Info Input
st.sidebar.header("Your Info (Optional)")
selected_state = st.sidebar.selectbox("Select your state:", sorted(df["State"].dropna().unique()))

# Sidebar group toggles
st.sidebar.subheader("👤 Customize Population Segments")
selected_races = st.sidebar.multiselect("Select race groups to simulate:", race_columns, default=race_columns)
selected_ages = st.sidebar.multiselect("Select age groups to simulate:", age_columns, default=age_columns)

# Features & Target
X = df.drop(columns=[
    "cases_per_100k", "estimated_positive_cases", "Percent_Positive",
    "Total_population", "Total_Specimens", "Week", "LandArea_SqMi", "Month", "State"
])
y = df["cases_per_100k"]

# Numeric Columns
numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols)
])
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42))
])
model.fit(X_train, y_train)

# Input Sliders
st.sidebar.header("Adjust Variables")
input_data = {}

for col in numerical_cols:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    default_val = float(df[col].mean())

    if col in selected_races or col in selected_ages:
        input_data[col] = st.sidebar.slider(col, min_value=min_val, max_value=max_val, value=default_val)
    else:
        input_data[col] = default_val

input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]

# Risk Level Classifier
def get_risk_level(value):
    if value < 10:
        return "Minimal", "✅"
    elif value < 25:
        return "Low", "🟡"
    elif value < 50:
        return "Moderate", "🟠"
    else:
        return "High", "🔴"

risk_level, emoji = get_risk_level(prediction)

# Results Section
st.markdown("### 📊 Prediction Results")
st.metric("Predicted cases per 100k", f"{prediction:.2f}")
st.markdown(f"### **Risk Level: {emoji} {risk_level}**")

# Optional Recommendations Section
st.markdown("---")
st.markdown("### 🛡️ Suggested Actions")
if risk_level == "High":
    st.warning("⚠️ High risk! Get vaccinated, wear a mask in public spaces, and avoid crowded places.")
elif risk_level == "Moderate":
    st.info("🧼 Moderate risk. Wash hands often and avoid large gatherings if possible.")
elif risk_level == "Low":
    st.success("😌 Low risk. Maintain good hygiene and monitor symptoms.")
else:
    st.success("✅ Minimal risk. Stay healthy and keep up good practices!")
