from sklearn.datasets import load_breast_cancer
import streamlit as st
import pandas as pd
import joblib
model = joblib.load("models/breast_cancer_model.pkl")
scaler = joblib.load("models/scaler.pkl")
cancer = load_breast_cancer()

df = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names
)
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide"
)
st.sidebar.title("📋 Project Information")

st.sidebar.markdown("""
### Breast Cancer Prediction

**Dataset**
- Wisconsin Breast Cancer Dataset

**Model**
- Logistic Regression

**Features**
- 30 Numerical Features

**Developer**
- Mustafa Shehata
""")
st.title("🩺 Breast Cancer Prediction App")

st.write(
    "Enter the tumor measurements below and click Predict."
)
feature_names = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness',
    'mean compactness',
    'mean concavity',
    'mean concave points',
    'mean symmetry',
    'mean fractal dimension',

    'radius error',
    'texture error',
    'perimeter error',
    'area error',
    'smoothness error',
    'compactness error',
    'concavity error',
    'concave points error',
    'symmetry error',
    'fractal dimension error',

    'worst radius',
    'worst texture',
    'worst perimeter',
    'worst area',
    'worst smoothness',
    'worst compactness',
    'worst concavity',
    'worst concave points',
    'worst symmetry',
    'worst fractal dimension'
]
user_input = []

col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):

    if i % 2 == 0:
        with col1:
            value = st.number_input(feature, value=float(df[feature].mean()))
    else:
        with col2:
            value = st.number_input(feature, value=float(df[feature].mean()))

    user_input.append(value)

if st.button("Predict"):

    input_df = pd.DataFrame([user_input], columns=feature_names)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.success("🟢 Benign")

    else:
        st.error("🔴 Malignant")

    confidence = probability.max()

    st.progress(confidence)

    st.success(
        f"Confidence: {confidence*100:.2f}%"
    )