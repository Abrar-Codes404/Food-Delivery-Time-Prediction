# 🍕 Food Delivery Time Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-red.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/)
[![GitHub](https://img.shields.io/badge/GitHub-Abrar--Codes404-black.svg?logo=github)](https://github.com/Abrar-Codes404)

An end-to-end Machine Learning regression application built with **Python**, **Scikit-learn**, and **Streamlit** to estimate food delivery duration in minutes based on delivery agent characteristics, environmental conditions, traffic density, vehicle type, order type, and delivery distance.

---

## 📌 Project Overview

Accurate delivery duration estimation is essential for logistics optimization, customer satisfaction, and courier dispatching in food delivery platforms. This project evaluates multiple machine learning regression algorithms, selects the best-performing model, encapsulates preprocessing and inference into a Scikit-learn Pipeline, and deploys an interactive web dashboard using Streamlit.

- **Dataset Origin**: Curated dataset from **Kaggle** (Kaggle Food Delivery Dataset).
- **Prediction Target**: Continuous **Delivery Duration in minutes** (`Delivery_Time_min`).
- **Champion Model**: **Ordinary Least Squares Linear Regression** (Lowest MAE: **8.49 min**, Lowest RMSE: **10.48 min**, Highest $R^2$: **0.65**).

---

## 🚀 Key Features & Highlights

- **Pre-trained Pipeline Integration**: Directly loads and executes `food_delivery_linear_regression.pkl` without retraining or manual feature preprocessing.
- **Reproducible Environment**: Uses pinned `scikit-learn==1.6.1` to ensure reliable deserialization and consistent prediction outputs.
- **Multi-Page Streamlit Dashboard**:
  1. 🏠 **Overview**: Hero section, KPI cards (~800 records, 10 features, Best $R^2$ = 0.65, Best MAE = 8.49 min), and 4-step workflow diagram.
  2. 🎯 **Delivery Time Prediction**: Interactive form with input validation, duration prediction, status interpretation badge, and downloadable CSV report.
  3. 📊 **Model Performance**: Structured evaluation metrics comparison across 5 algorithms, 3-panel Plotly comparison chart, and Actual vs Predicted scatter plot ($y = x$).
  4. 🔍 **Feature Analysis**: Interactive horizontal Plotly bar chart showing top fitted Linear Regression feature coefficients with statistical explanations.
  5. 📋 **Dataset & Methodology**: End-to-end preprocessing flow, imputation + One-Hot Encoding details, and 7 key findings.
  6. 👨‍💻 **About the Project**: Tech stack details, project objectives, model technical health info, and model selection rationale.

---

## 📊 Dataset & Features (Sourced from Kaggle)

- **Records**: 794 cleaned operational delivery records
- **Target Variable**: `Delivery_Time_min` (Continuous numerical target in minutes)
- **Input Features (10 total)**:
  - **4 Numerical**: `Delivery_Person_Age`, `Delivery_Person_Rating`, `Multiple_Deliveries`, `Distance_km`
  - **6 Categorical**: `Weather`, `Traffic_Level`, `Vehicle_Type`, `Area_Type`, `Order_Type`, `Festival`

---

## 📈 Preprocessing & ML Pipeline

The trained model is encapsulated in a Scikit-Learn `Pipeline`:

```
Raw Data Input
    ↓
ColumnTransformer
    ├── Numerical Pipeline: SimpleImputer(strategy='median')
    └── Categorical Pipeline: SimpleImputer(strategy='most_frequent') 
                             → OneHotEncoder(handle_unknown='ignore')
    ↓
LinearRegression() Model
```

---

## 🏆 Model Evaluation Benchmark

| Model Architecture | MAE (min) | MSE | RMSE (min) | R² Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (OLS)** | **8.49** | **109.92** | **10.48** | **0.65** | **Champion Model** |
| Gradient Boosting | 8.66 | 113.04 | 10.63 | 0.64 | Candidate |
| Random Forest | 9.13 | 124.71 | 11.17 | 0.61 | Candidate |
| Tuned Random Forest | 9.22 | 125.25 | 11.19 | 0.60 | Candidate |
| Decision Tree | 11.61 | 209.89 | 14.49 | 0.34 | Candidate |

**Selected Model Rationale:** Linear Regression achieved the lowest Mean Absolute Error (**8.49 minutes**) and highest $R^2$ (**0.65**) among all evaluated algorithms while offering high interpretability and minimal computational overhead.

---

## 🛠️ Project Structure

```
Food-Delivery-Time-Prediction/
│
├── app.py                             # Main Streamlit web application & UI
├── model_results.py                   # Verified evaluation metrics dictionary & metadata
├── utils.py                           # Model loader, prediction helper, CSV report exporter
├── test_predictions.csv               # Dataset predictions for Actual vs Predicted visualization
├── cleaned_food_delivery_dataset (1).csv # Validated Kaggle food delivery dataset
├── food_delivery_linear_regression.pkl # Pre-trained Scikit-Learn Pipeline
├── requirements.txt                   # Pinned project dependencies
├── screenshots/                       # Application UI screenshots
│   ├── 1_Overview.png
│   ├── 2_Prediction.png
│   ├── 3_Model_Performance.png
│   ├── 4_Feature_Analysis.png
│   ├── 5_Dataset_Methodology.png
│   └── 6_About_Project.png
└── README.md                          # Project documentation
```

---

## 💻 Local Setup & Running Instructions

### 1. Clone Repository & Navigate to Directory
```bash
git clone https://github.com/Abrar-Codes404/Food-Delivery-Time-Prediction.git
cd Food-Delivery-Time-Prediction
```

### 2. Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Streamlit Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## ☁️ Deployment to Streamlit Community Cloud (Free)

1. Fork or push this repository to your GitHub account: `Abrar-Codes404/Food-Delivery-Time-Prediction`.
2. Visit **[share.streamlit.io](https://share.streamlit.io/)**.
3. Sign in with GitHub and click **"Create app"**.
4. Select:
   - **Repository:** `Abrar-Codes404/Food-Delivery-Time-Prediction`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"**. Streamlit Cloud will install dependencies and host your live web app with a public URL.

---

## 👨‍💻 Author

- **GitHub**: [@Abrar-Codes404](https://github.com/Abrar-Codes404)
- **Project**: Food Delivery Time Prediction Using Machine Learning
