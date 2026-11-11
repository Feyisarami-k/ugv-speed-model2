import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page layout and title
st.set_page_config(page_title="UGV Speed Predictor", layout="centered")
st.title("🚗 Unmanned Ground Vehicle (UGV) Speed Predictor")
st.write("""
This app predicts the **Average Speed (m/s)** of a UGV based on vehicle specs and environmental parameters.
""")

# Load the saved model and scaler
@st.cache_resource
def load_assets():
    with open("ugv_gb.pkl", "rb") as f:
        model = pickle.load(f)
    with open("ugv_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("Error: Model or Scaler files not found! Make sure 'ugv_gb.pkl' and 'ugv_scaler.pkl' are in the same folder.")
    st.stop()

# Sidebar Inputs for features
st.sidebar.header("Configuration & Environmental Inputs")

# Categorical inputs
vehicle_type = st.sidebar.selectbox("Vehicle Type", ["4x4 wheeled", "6x6 wheeled", "tracked"])
terrain = st.sidebar.selectbox("Terrain", ["rock", "sand", "gravel", "mud", "paved", "snow"])
operating_mode = st.sidebar.selectbox("Operating Mode", ["teleoperated", "autonomous", "convoy-follow"])

# Numerical inputs
vehicle_mass_t = st.sidebar.slider("Vehicle Mass (tons)", 1.0, 20.0, 8.0, 0.1)
payload_kg = st.sidebar.slider("Payload (kg)", 0.0, 1500.0, 500.0, 10.0)
slope = st.sidebar.slider("Slope (Degrees)", -20.0, 50.0, 5.0, 0.1)
cone_index = st.sidebar.slider("Cone Index", 100.0, 2000.0, 800.0, 10.0)
visibility = st.sidebar.slider("Visibility (meters)", 0.0, 5000.0, 1000.0, 50.0)
ambient_temp_c = st.sidebar.slider("Ambient Temp (°C)", -10.0, 50.0, 25.0, 0.5)
obstacles_per_100m = st.sidebar.slider("Obstacles per 100m", 0.0, 30.0, 5.0, 0.1)
latency = st.sidebar.slider("Latency (ms)", 0.0, 400.0, 100.0, 5.0)
battery_pct = st.sidebar.slider("Battery Percentage (%)", 0.0, 100.0, 80.0, 1.0)
slip = st.sidebar.slider("Slip (%)", 0.0, 40.0, 25.0, 0.1)

# Organize user inputs into a DataFrame template mimicking training columns
input_data = pd.DataFrame({
    'vehicle_mass_t': [vehicle_mass_t],
    'payload_kg': [payload_kg],
    'Slope': [slope],
    'Cone Index': [cone_index],
    'Visibility': [visibility],
    'ambient_temp_c': [ambient_temp_c],
    'Obstacles/100m': [obstacles_per_100m],
    'Latency': [latency],
    'battery_pct': [battery_pct],
    'Slip': [slip],

    # Initialize dummy variables to False
    'vehicle_type_tracked': [False],
    'terrain_gravel': [False],
    'terrain_mud': [False],
    'terrain_paved': [False],
    'terrain_rock': [False],
    'terrain_sand': [False],
    'terrain_snow': [False],
    'operating_mode_autonomous': [False],
    'operating_mode_convoy-follow': [False],
    'operating_mode_teleoperated': [False]
})

# Apply user's selected categorical values to match One-Hot encoding
if vehicle_type == 'tracked':
    input_data['vehicle_type_tracked'] = True

if terrain != 'a_not_known':
    terrain_col = f"terrain_{terrain}"
    if terrain_col in input_data.columns:
        input_data[terrain_col] = True

if operating_mode != 'a_not_known':
    mode_col = f"operating_mode_{operating_mode}"
    if mode_col in input_data.columns:
        input_data[mode_col] = True

# List of continuous columns to scale
scaled_cols = [
    'vehicle_mass_t', 'payload_kg', 'Slope', 'Cone Index',
    'Visibility', 'ambient_temp_c', 'Obstacles/100m',
    'Latency', 'battery_pct', 'Slip'
]

# Process and display results
if st.button("Predict UGV Speed"):
    try:
        # Scale continuous numerical features using the loaded scaler
        input_scaled = input_data.copy()
        input_scaled[scaled_cols] = scaler.transform(input_data[scaled_cols])

        # Generate prediction
        prediction = model.predict(input_scaled)[0]

        st.success(f"### ⏱️ Predicted Average Speed: **{prediction:.2f} m/s**")
        st.metric(label="Estimated Speed", value=f"{prediction:.2f} m/s")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
