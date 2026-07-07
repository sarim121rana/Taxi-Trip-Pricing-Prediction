import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------------------------------
# 1. Load the Saved Model and Scaler
# -----------------------------------------------------------------------------
# We use st.cache_resource so these only load once when the app starts
@st.cache_resource
def load_assets():
    model = joblib.load('taxi_price_model.pkl')
    scaler = joblib.load('taxi_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# -----------------------------------------------------------------------------
# 2. Build the Streamlit UI
# -----------------------------------------------------------------------------
st.title("🚖 Taxi Trip Price Predictor")
st.write("Enter the trip details below to get an estimated fare.")

# Create columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("Trip Distance (km)", min_value=0.1, max_value=200.0, value=10.0, step=0.5)
    duration = st.number_input("Trip Duration (minutes)", min_value=1.0, max_value=300.0, value=25.0, step=1.0)
    passengers = st.selectbox("Passenger Count", [1.0, 2.0, 3.0, 4.0])
    
    # Financial metrics (you can set defaults based on your dataset's mean)
    base_fare = st.number_input("Base Fare", min_value=0.0, value=3.50)

with col2:
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
    day_type = st.selectbox("Day of Week", ["Weekday", "Weekend"])
    traffic = st.selectbox("Traffic Conditions", ["Low", "Medium", "High"])
    weather = st.selectbox("Weather", ["Clear", "Rain", "Snow"])
    
    per_km_rate = st.number_input("Per Km Rate", min_value=0.0, value=1.20)
    per_minute_rate = st.number_input("Per Minute Rate", min_value=0.0, value=0.25)

# -----------------------------------------------------------------------------
# 3. Process the Input Data
# -----------------------------------------------------------------------------
# Dictionary mappings to match our Label Encoding from the notebook
time_map = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
traffic_map = {'Low': 0, 'Medium': 1, 'High': 2}

if st.button("Predict Price"):
    # Convert categorical inputs to the numerical format the model expects
    time_encoded = time_map[time_of_day]
    traffic_encoded = traffic_map[traffic]
    
    # Manual One-Hot Encoding logic for Day and Weather
    is_weekend = 1 if day_type == "Weekend" else 0
    is_rain = 1 if weather == "Rain" else 0
    is_snow = 1 if weather == "Snow" else 0

    # Create a DataFrame with a single row of the user's input
    # IMPORTANT: The columns MUST match the exact order and names from training!
    input_data = pd.DataFrame({
        'Trip_Distance_km': [distance],
        'Time_of_Day': [time_encoded],
        'Passenger_Count': [passengers],
        'Traffic_Conditions': [traffic_encoded],
        'Base_Fare': [base_fare],
        'Per_Km_Rate': [per_km_rate],
        'Per_Minute_Rate': [per_minute_rate],
        'Trip_Duration_Minutes': [duration],
        'Day_of_Week_Weekend': [is_weekend],
        'Weather_Rain': [is_rain],
        'Weather_Snow': [is_snow]
    })

    # Apply the StandardScaler ONLY to the columns we scaled during training
    cols_to_scale = ['Trip_Distance_km', 'Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes']
    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

    # -----------------------------------------------------------------------------
    # 4. Make the Prediction
    # -----------------------------------------------------------------------------
    prediction = model.predict(input_data)[0]
    
    # Display the result
    st.success(f"### Estimated Trip Price: ${prediction:.2f}")
    st.balloons()