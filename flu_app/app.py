import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load data from GitHub
@st.cache_data
def load_data():
    return pd.read_csv("flu_app/flu_data.csv")

df = load_data()

st.title("Flu Outbreak Risk Predictor")
st.write("This app predicts flu cases per 100k people and determines outbreak risk level.")

# Define categorical columns manually (update if needed)
categorical_cols = ["race", "Sex"]

# Define X and y
X = df.drop(columns=[
    "cases_per_100k", "estimated_positive_cases", "Percent_Positive",
    "Total_population", "Total_Specimens", "Week", "LandArea_SqMi",
    "Month", "State"
])
y = df["cases_per_100k"]

# Detect numerical columns
numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns.difference(categorical_cols)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols)
])

# Final model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42))
])

# Train model
model.fit(X_train, y_train)

# User input
st.sidebar.header("Input Variables")

input_data = {}
for col in numerical_cols:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    default_val = float(df[col].mean())
    input_data[col] = st.sidebar.slider(col, min_value=min_val, max_value=max_val, value=default_val)

input_df = pd.DataFrame([input_data])

# Prediction
prediction = model.predict(input_df)[0]

# Risk level logic
def get_risk_level(value):
    if value < 10:
        return "Minimal"
    elif value < 25:
        return "Low"
    elif value < 50:
        return "Moderate"
    else:
        return "High"

risk_level = get_risk_level(prediction)

# Output
st.subheader("Prediction Results")
st.write(f"**Predicted cases per 100k:** `{prediction:.2f}`")

# Risk display with styling
if risk_level == "Minimal":
    st.success(f"Risk level: {risk_level} 🟢")
elif risk_level == "Low":
    st.info(f"Risk level: {risk_level} 🟡")
elif risk_level == "Moderate":
    st.warning(f"Risk level: {risk_level} 🟠")
else:
    st.error(f"Risk level: {risk_level} 🔴")

# Mitigation strategies
st.subheader("Suggested Flu Prevention Steps")

mitigation = {
    "Minimal": "✅ Stay informed and maintain good hygiene habits.",
    "Low": "😷 Wash hands often, avoid sharing personal items, and monitor symptoms.",
    "Moderate": "🧼 Increase handwashing, wear a mask in crowded areas, and reduce indoor gatherings.",
    "High": "🚨 Limit exposure to crowds, wear a mask in public, stay home when possible, and consider getting a flu shot."
}

st.markdown(mitigation[risk_level])
