# 🚖 Taxi Trip Price Predictor

## 📌 Project Overview
The **Taxi Trip Price Predictor** is an end-to-end Machine Learning regression project designed to estimate the total fare of a taxi trip based on various journey details. The project progresses from raw data cleaning and exploratory data analysis (EDA) to building, optimizing, and deploying a high-performing Random Forest regression model via a Streamlit web application.
## APP Link (https://taxi-trip-pricing-prediction-36wc7zfloyxpzmxrtfhr58.streamlit.app/)
## ⚙️ Tech Stack
* **Language:** Python
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Linear Regression, Random Forest Regressor)
* **Web Framework:** Streamlit
* **Model Serialization:** Joblib

## 🛠️ Project Workflow

1. **Data Cleaning & Imputation:**
   * Handled missing (NaN) values across 10+ columns.
   * Filled continuous features (e.g., `Base_Fare`, `Trip_Duration_Minutes`) with mean values.
   * Filled categorical features (e.g., `Weather`, `Traffic_Conditions`) with mode values.
2. **Feature Engineering & Encoding:**
   * Applied **Label Encoding** for ordinal data (`Traffic_Conditions`, `Time_of_Day`).
   * Applied **One-Hot Encoding** for nominal data (`Weather`, `Day_of_Week`) while avoiding the dummy variable trap.
3. **Exploratory Data Analysis (EDA):**
   * Generated correlation heatmaps to identify strong predictors (e.g., `Trip_Distance_km`).
   * Visualized distributions, relationships, and outliers using Seaborn scatter plots and boxplots.
4. **Data Scaling:**
   * Standardized continuous variables using `StandardScaler` to ensure unbiased model training.
5. **Model Training & Evaluation:**
   * Trained a baseline **Multiple Linear Regression** model ($R^2$: ~0.70).
   * Upgraded to a **Random Forest Regressor** to capture non-linear relationships, significantly improving accuracy ($R^2$: ~0.92).

## 📊 Model Performance
| Model | Mean Absolute Error (MAE) | R-Squared ($R^2$) |
| :--- | :--- | :--- |
| **Linear Regression (Baseline)** | 12.35 | 0.697 |
| **Random Forest Regressor (Final)** | **5.82** | **0.919** |

## 📁 Repository Structure
```text
├── Taxi Trip pricing.ipynb    # Original Jupyter Notebook with full ML pipeline
├── app.py                     # Streamlit web application script
├── requirements.txt           # Python dependencies required to run the app
├── taxi_price_model.pkl       # Serialized Random Forest model
├── taxi_scaler.pkl            # Serialized StandardScaler for input processing
└── README.md                  # Project documentation
