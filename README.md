# Heart-Disease-Prediction-ANN
Heart Disease Prediction using ANN 

## 🌐 Live Demo
👉 **[heart-disease-prediction-annmodel.streamlit.app](https://heart-disease-prediction-annmodel.streamlit.app/)**

---

## 📌 Project Overview
This project focuses on predicting the presence of heart disease in patients using a **Deep Learning Artificial Neural Network (ANN)** built with **TensorFlow and Keras**.

The aim is to build an end-to-end deep learning pipeline starting from data preprocessing to model deployment using a Streamlit web application.

The project demonstrates practical skills in:
- Data cleaning & preprocessing
- Feature encoding & scaling
- Building & training a deep learning ANN model
- Model evaluation with multiple metrics
- Model deployment with an interactive UI

---

## 📂 Dataset
- **Dataset Type:** Tabular medical dataset
- **Target Variable:** Heart Disease Status
- **Problem Type:** Binary Classification
- **Classes:**
  - `0 / False` → No Heart Disease
  - `1 / True` → Heart Disease

The dataset contains patient information such as age, blood pressure, cholesterol levels, BMI, sleep hours, and other health indicators.

---

## 🛠️ Technologies & Libraries Used
- **Python**
- **Pandas & NumPy** – Data handling
- **Matplotlib & Seaborn** – Visualization
- **TensorFlow & Keras** – Deep learning model building & training
- **Scikit-learn** – Preprocessing & evaluation
- **Streamlit** – Web application
- **Pickle** – Model serialization

---

## 🔄 Project Workflow

**1️⃣ Data Exploration & Profiling**
- Used `head()`, `info()`, `describe()`, and null checks
- Visualized target distribution and age distribution
- Plotted feature correlation heatmap

**2️⃣ Data Cleaning**
- Handled missing values (mode for categorical, mean for numerical)
- Ensured correct data types

**3️⃣ Feature Engineering**
- Target encoding using `LabelEncoder`
- Selected 5 key features: Age, Blood Pressure, Cholesterol, BMI, Sleep Hours

**4️⃣ Feature Scaling**
- Applied `StandardScaler` to normalize input features

**5️⃣ Model Building — ANN Architecture**
