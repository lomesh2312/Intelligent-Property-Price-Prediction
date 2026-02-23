import streamlit as st
import pandas as pd
import joblib

# Load model
model, model_name = joblib.load("model/best_house_price_model.pkl")

st.title("🏠 Intelligent Property Price Prediction System")

st.header("Enter Property Details")

area = st.number_input("Area (sq ft)", min_value=1650, max_value=16200, value= 1650)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=4, value=2)
stories = st.number_input("Stories", min_value=1, max_value=4, value=2)
parking = st.number_input("Parking Spaces", min_value=0, max_value=3, value=1)

guestroom = st.selectbox("Guest Room", ["No", "Yes"])
mainroad = st.selectbox("Main Road Access", ["No", "Yes"])
prefarea = st.selectbox("Preferred Area", ["No", "Yes"])
basement = st.selectbox("Basement", ["No", "Yes"])
airconditioning = st.selectbox("Air Conditioning", ["No", "Yes"])

furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

if st.button("Predict Price"):

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