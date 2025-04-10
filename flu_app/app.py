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
st.title("🦠 Flu Outbreak Risk Predictor")
st.write("This app predicts flu cases per 100k people and determines outbreak risk level.")

# Sidebar – Personal Info Input
st.sidebar.header("Your Info (Optional)")
selected_state = st.sidebar.selectbox("Select your state:", sorted(df["State"].dropna().unique()))
selected_race = st.sidebar.selectbox("Select your race:", sorted(df["race"].dropna().unique()))

# Features & Target
categorical_cols = ["race"]  # "Sex" removed as it's not a column
X = df.drop(columns=[
    "cases_per_100k", "estimated_positive_cases", "Percent_Positive",
    "Total_population", "Total_Specimens", "Week", "LandArea_SqMi", "Month", "State"
])
y = df["cases_per_100k"]

# Numeric Columns
numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns.difference(categorical_cols)

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
    input_data[col] = st.sidebar.slider(col, min_value=min_val, max_value=max_val, value=default_val)

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
st.markdown(f"**Risk Level:** {emoji} `{risk_level}`")

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

