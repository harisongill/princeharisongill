import streamlit as st
import json
import joblib
import numpy as np
import pandas as pd

# Loading schema
with open("schema.json", "r") as f:
    schema = json.load(f)

# Load model
model = joblib.load("model_pipeline.pkl")

st.title("Employee Attrition Predictor")
st.subheader("Created by Prince Harison Gill")

st.write("Please fill in the employee details below:")

# Gather user input
input_data = {}
for col in schema["numerical_features"]:
    input_data[col] = st.number_input(f"{col}", value=0)

# Update this list based on your dataset values
categorical_options = {
    "Education": ["Bachelors", "Masters", "PhD"],
    "City": ["New York", "Los Angeles", "Chicago"],
    "Gender": ["Male", "Female"],
    "EverBenched": ["Yes", "No"]
}

for col in schema["categorical_features"]:
    options = categorical_options.get(col, ["Option1", "Option2"])
    input_data[col] = st.selectbox(f"{col}", options)

# Predict
if st.button("Submit"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    result = "Yes" if prediction == 1 else "No"
    st.success(f"Prediction: Will the employee leave? → **{result}**")
