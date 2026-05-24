# Heart-Disease-Prediction-ANN
Heart Disease Prediction using ANN 

---

##  Project Overview
This project focuses on predicting the presence of heart disease in patients using a **Deep Learning Artificial Neural Network (ANN)** built with **TensorFlow and Keras**.

The aim is to build an end-to-end deep learning pipeline starting from data preprocessing to model deployment using a Streamlit web application.

The project demonstrates practical skills in:
- Data cleaning & preprocessing
- Feature encoding & scaling
- Building & training a deep learning ANN model
- Model evaluation with multiple metrics
- Model deployment with an interactive UI

---

##  Dataset
- **Dataset Type:** Tabular medical dataset
- **Target Variable:** Heart Disease Status
- **Problem Type:** Binary Classification
- **Classes:**
  - `0 / False` → No Heart Disease
  - `1 / True` → Heart Disease

The dataset contains patient information such as age, blood pressure, cholesterol levels, BMI, sleep hours, and other health indicators.

---

##  Technologies & Libraries Used
- **Python**
- **Pandas & NumPy** – Data handling
- **Matplotlib & Seaborn** – Visualization
- **TensorFlow & Keras** – Deep learning model building & training
- **Scikit-learn** – Preprocessing & evaluation
- **Streamlit** – Web application
- **Pickle** – Model serialization

---

##  Project Workflow

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
Input (5 features)
↓
Dense(64, ReLU) → BatchNormalization → Dropout(0.3)
↓
Dense(32, ReLU) → BatchNormalization → Dropout(0.2)
↓
Dense(16, ReLU)
↓
Dense(1, Sigmoid) → Output Probability

**6️⃣ Model Training**
- Optimizer: Adam (lr=0.001)
- Loss: Binary Cross-Entropy
- Epochs: Up to 150
- Batch Size: 32
- Callbacks: EarlyStopping + ReduceLROnPlateau

**7️⃣ Model Evaluation**
- Models were evaluated using:
  - Accuracy Score
  - Classification Report
  - Confusion Matrix
  - Prediction Probability Distribution
  - Training vs Validation Loss & Accuracy Curves

**8️⃣ Model Saving**
- Trained model saved using `pickle`
- Scaler saved separately to ensure consistent preprocessing during prediction

---

##  Model Performance
The ANN model achieved reliable accuracy on the test dataset.
Evaluation metrics such as classification report, confusion matrix, and probability distribution plots were used to analyze performance beyond accuracy.

Advanced techniques like **BatchNormalization**, **Dropout**, **EarlyStopping**, and **ReduceLROnPlateau** were applied to improve generalization and prevent overfitting.

---

##  Web Application — Streamlit

An interactive Streamlit web app was developed to:
- Take user health inputs via sliders
- Apply the same preprocessing steps used during training
- Predict heart disease risk in real time
- Display risk probability and health recommendation

### 👉 Live Demo
**[heart-disease-prediction-annmodel.streamlit.app](https://heart-disease-prediction-annmodel.streamlit.app/)**
