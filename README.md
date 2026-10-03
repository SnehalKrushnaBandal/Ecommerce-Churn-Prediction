# 🛒 E-Commerce Customer Churn Prediction

A machine learning-based web application that predicts whether an e-commerce customer is likely to churn based on their behavioral and demographic information.

The project compares multiple machine learning classification algorithms and deploys the selected model using Streamlit.

---

## 📌 Abstract

Customer churn is an important challenge for e-commerce businesses because losing existing customers can negatively affect revenue and long-term customer relationships.

This project develops a machine learning-based customer churn prediction system using an e-commerce customer dataset. Exploratory Data Analysis (EDA), data cleaning, preprocessing, feature encoding, and classification techniques are applied to identify patterns associated with customer churn.

Three machine learning models were evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting

The models were evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrices, classification reports, and ROC curves.

Based on the test-set results, Random Forest achieved the strongest measured performance among the three evaluated models and was selected as the final prediction model.

The trained model is integrated into a Streamlit web application where users can enter customer information and receive a churn prediction along with the estimated model probability.

---

## 🎯 Problem Statement

E-commerce companies need to identify customers who may stop using their services so that appropriate retention strategies can be considered.

The objective of this project is to develop a machine learning system that predicts customer churn using customer behavioral, transactional, and demographic information.

---

## 🎯 Objectives

- Analyze an e-commerce customer dataset.
- Perform Exploratory Data Analysis (EDA).
- Identify and handle missing values.
- Analyze relationships between customer characteristics and churn.
- Prepare categorical and numerical features for machine learning.
- Train multiple classification models.
- Compare model performance using standard evaluation metrics.
- Select a suitable final model based on measured test-set performance.
- Deploy the trained model using Streamlit.
- Provide an interactive customer churn prediction interface.

---

## 📊 Dataset

The project uses the **E-Commerce Customer Churn Dataset**.

The dataset contains **5,630 customer records** and **20 columns**.

Important features include:

- CustomerID
- Churn
- Tenure
- PreferredLoginDevice
- CityTier
- WarehouseToHome
- PreferredPaymentMode
- Gender
- HourSpendOnApp
- NumberOfDeviceRegistered
- PreferedOrderCat
- SatisfactionScore
- MaritalStatus
- NumberOfAddress
- Complain
- OrderAmountHikeFromlastYear
- CouponUsed
- OrderCount
- DaySinceLastOrder
- CashbackAmount

`CustomerID` is used only as an identifier and is excluded from model training.

The target variable is:

- `Churn = 1` → Customer churned
- `Churn = 0` → Customer did not churn

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn

### Model Storage

- Joblib

### Web Application

- Streamlit

### Development Environment

- VS Code
- Jupyter Notebook
- uv

---

## 📁 Project Structure

```text
Ecommerce-Churn-Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── E Commerce Dataset.xlsx
│
├── images/
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│   └── 01_EDA_and_Preprocessing.ipynb
│
├── src/
│
├── .gitignore
├── README.md
├── pyproject.toml
└── uv.lock