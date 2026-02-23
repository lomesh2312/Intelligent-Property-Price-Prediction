import streamlit as st
import pandas as pd
import joblib

import os

model_path = os.path.join(os.path.dirname(__file__), "model", "best_house_price_model.pkl")
model, model_name = joblib.load(model_path)
st.title("🏠 Intelligent Property Price Prediction System")

st.header("Enter Property Details")

area = st.number_input("Area (sq ft)", min_value=1650, max_value=16200, placeholder="Enter area in sq ft greater than 1650")
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, placeholder="Enter number of bedrooms (1-6)")
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=4, placeholder="Enter number of bathrooms (1-4)")
stories = st.number_input("Stories", min_value=1, max_value=4, placeholder="Enter number of stories (1-4)")
parking = st.number_input("Parking Spaces", min_value=0, max_value=3, placeholder="Enter number of parking spaces (0-3)")

guestroom = st.selectbox(
    "Guest Room",
    ["Select...", "No", "Yes"],
    index=0
)

mainroad = st.selectbox(
    "Main Road Access",
    ["Select...", "No", "Yes"],
    index=0
)

prefarea = st.selectbox(
    "Preferred Area",
    ["Select...", "No", "Yes"],
    index=0
)

basement = st.selectbox(
    "Basement",
    ["Select...", "No", "Yes"],
    index=0
)

airconditioning = st.selectbox(
    "Air Conditioning",
    ["Select...", "No", "Yes"],
    index=0
)

furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["Select...", "furnished", "semi-furnished", "unfurnished"],
    index=0
)

if st.button("Predict Price"):
    if "Select..." in [
        guestroom, mainroad, prefarea,
        basement, airconditioning, furnishingstatus
    ]:
        st.error("⚠ Please select all dropdown options before predicting.")
        st.stop()

    if area <= 0 or bedrooms <= 0 or bathrooms <= 0:
        st.error("⚠ Please enter valid numeric values.")
        st.stop()

    semi = 1 if furnishingstatus == "semi-furnished" else 0
    unfurnished = 1 if furnishingstatus == "unfurnished" else 0

    input_data = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "guestroom": 1 if guestroom == "Yes" else 0,
        "bathrooms": bathrooms,
        "mainroad": 1 if mainroad == "Yes" else 0,
        "prefarea": 1 if prefarea == "Yes" else 0,
        "stories": stories,
        "parking": parking,
        "basement": 1 if basement == "Yes" else 0,
        "airconditioning": 1 if airconditioning == "Yes" else 0,
        "furnishingstatus_semi-furnished": semi,
        "furnishingstatus_unfurnished": unfurnished
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Price: ₹ {round(prediction, 2)}")
    st.info(f"📊 Model Used: {model_name}")