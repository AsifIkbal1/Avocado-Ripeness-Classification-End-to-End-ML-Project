import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 🌐 Load the saved model and scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("regression.pkl", "rb") as f:
    model = pickle.load(f)

# 🏷️ Color categories (use the same LabelEncoder order used during training)
color_categories = ['black', 'green', 'dark green', 'purple']  # Modify based on your data

# 🌟 Streamlit UI
st.title("🥑 Avocado Ripeness Predictor")
st.write("Input avocado features below to predict ripeness category.")

# 🎛️ Input features
firmness = st.slider("Firmness", min_value=0.0, max_value=100.0, value=50.0)
hue = st.slider("Hue", min_value=0, max_value=360, value=180)
saturation = st.slider("Saturation", min_value=0, max_value=100, value=50)
brightness = st.slider("Brightness", min_value=0, max_value=100, value=50)
color_category = st.selectbox("Color Category", color_categories)
sound_db = st.slider("Sound dB", min_value=0, max_value=100, value=50)
weight_g = st.slider("Weight (g)", min_value=100, max_value=500, value=200)
size_cm3 = st.slider("Size (cm³)", min_value=100, max_value=500, value=250)

# 🧮 Encode color category (match your LabelEncoder used in training)
color_map = {name: idx for idx, name in enumerate(color_categories)}
color_encoded = color_map[color_category]

# 🧩 Create input array and scale
input_data = np.array([[firmness, hue, saturation, brightness, color_encoded,
                        sound_db, weight_g, size_cm3]])
input_scaled = scaler.transform(input_data)

# 🔮 Predict
if st.button("Predict Ripeness"):
    prediction = model.predict(input_scaled)
    
    # 🎯 Decode ripeness label if needed
    ripeness_labels = ['breaking', 'hard', 'pre-conditioned', 'ripe']  # example order
    try:
        pred_label = ripeness_labels[int(prediction[0])]
    except:
        pred_label = str(prediction[0])

    st.success(f"Predicted Ripeness: **{pred_label}**")
