import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(page_title="Taxi Fare Predictor", page_icon="🚖", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; }
    </style>
    """, unsafe_allow_html=True)

# 1. Load Assets
@st.cache_resource
def load_assets():
    model = joblib.load('taxi_price_model.pkl')
    scaler = joblib.load('taxi_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 2. Sidebar UI
st.sidebar.header("📍 Trip Parameters")
distance = st.sidebar.slider("Trip Distance (km)", 0.1, 100.0, 10.0)
duration = st.sidebar.slider("Trip Duration (minutes)", 1.0, 300.0, 25.0)
passengers = st.sidebar.selectbox("Passengers", [1, 2, 3, 4])

st.sidebar.header("⚙️ Conditions")
time_of_day = st.sidebar.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
day_type = st.sidebar.selectbox("Day Type", ["Weekday", "Weekend"])
traffic = st.sidebar.selectbox("Traffic", ["Low", "Medium", "High"])
weather = st.sidebar.selectbox("Weather", ["Clear", "Rain", "Snow"])

# 3. Main Interface
st.title("🚖 Taxi Fare Prediction AI")
st.markdown("Enter the trip details in the **sidebar** to get a fair estimate.")

if st.button("🚀 Predict Fare"):
    # Encoding logic
    time_map = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    traffic_map = {'Low': 0, 'Medium': 1, 'High': 2}
    
    # Input DataFrame
    input_data = pd.DataFrame({
        'Trip_Distance_km': [distance],
        'Time_of_Day': [time_map[time_of_day]],
        'Passenger_Count': [float(passengers)],
        'Traffic_Conditions': [traffic_map[traffic]],
        'Base_Fare': [3.50],
        'Per_Km_Rate': [1.20],
        'Per_Minute_Rate': [0.25],
        'Trip_Duration_Minutes': [duration],
        'Day_of_Week_Weekend': [1 if day_type == "Weekend" else 0],
        'Weather_Rain': [1 if weather == "Rain" else 0],
        'Weather_Snow': [1 if weather == "Snow" else 0]
    })

    # Transform & Predict
    cols_to_scale = ['Trip_Distance_km', 'Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes']
    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])
    
    prediction = model.predict(input_data)[0]
    
    # Display Result
    st.metric(label="Estimated Price", value=f"₹{prediction:,.2f}")
    st.balloons()
