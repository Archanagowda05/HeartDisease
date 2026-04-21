# 🫀 Heart Disease Prediction Dashboard

An end-to-end Machine Learning project for predicting heart disease risk using patient clinical data and an interactive Streamlit dashboard for visualization and risk assessment.

## Overview

This project analyzes cardiovascular health indicators and predicts whether a patient is at risk of heart disease using a Random Forest Classifier.

The application combines:

- Data preprocessing  
- Exploratory Data Analysis (EDA)  
- Machine Learning model training  
- Risk prediction  
- Interactive dashboard development  
- Visualization and feature analysis  

---

## Problem Statement

Heart disease is one of the leading causes of death worldwide. Early risk identification can support preventive care and faster medical intervention.

This project aims to build a predictive system that estimates heart disease risk using patient health parameters.

---

## Dataset

Dataset: Heart Disease Dataset

Features used:

- Age  
- Sex  
- Chest Pain Type (cp)  
- Resting Blood Pressure (trestbps)  
- Serum Cholesterol (chol)  
- Fasting Blood Sugar (fbs)  
- Resting ECG (restecg)  
- Maximum Heart Rate (thalach)  
- Exercise Induced Angina (exang)  
- Oldpeak  
- ST Slope (slope)  
- Number of Major Vessels (ca)  
- Thalassemia (thal)  

Target:

- 0 → No Heart Disease  
- 1 → Heart Disease Risk

---

## Machine Learning Model

Algorithm Used:

Random Forest Classifier

Why Random Forest?

- Handles tabular medical data well  
- Reduces overfitting  
- Good classification performance  
- Provides feature importance scores  
- Robust for structured datasets  

Model Performance:

Accuracy: 98.5%

---

## Dashboard Features

### Patient Risk Prediction
Interactive patient inputs:

- Age
- Blood Pressure
- Cholesterol
- Heart Rate
- Chest Pain Type

Predicts:

- Low Risk
- High Heart Disease Risk

---

## Data Visualizations

The dashboard includes:

### Disease Distribution
Shows count of patients with and without heart disease.

### Age vs Disease Analysis
Relationship between age and heart disease risk.

### Cholesterol Distribution
Analyzes cholesterol spread among patients.

### Feature Importance
Displays which clinical variables influence prediction most.

### Correlation Heatmap
Shows correlations among medical features.

---

## Tech Stack

Python

Libraries Used:

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

Deployment:

- Streamlit
- ngrok / Colab testing

Version Control:

- Git
- GitHub

---

## Project Workflow

1. Load Dataset  
2. Data Exploration  
3. Train Random Forest Model  
4. Evaluate Accuracy  
5. Build Streamlit Dashboard  
6. Add Interactive Prediction Inputs  
7. Add Visual Analytics  
8. Deploy and Push to GitHub

---

## Results

- Successfully trained prediction model
- Achieved high classification accuracy
- Built interactive medical risk dashboard
- Integrated prediction + visualization in one application

---

## Future Improvements

Possible enhancements:

- Compare multiple ML models
  - Logistic Regression
  - SVM
  - XGBoost

- Add:

  - Risk probability score
  - SHAP explainability
  - Better medical UI/UX
  - Real-time patient form
  - Cloud deployment

---

## Repository Structure

```bash
HeartDisease/
│
├── app.py
├── heart.csv
├── HeartDisease.ipynb
└── README.md
```

---

## Run Locally

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn

streamlit run app.py
```

---

## Disclaimer

This project is for educational and research purposes only.

It is not a substitute for professional medical diagnosis.

---

## Author

Archana Gowda

GitHub:
https://github.com/Archanagowda05
