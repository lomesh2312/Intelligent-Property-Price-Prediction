import os
import joblib
import pandas as pd
from langchain_groq import ChatGroq

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "..", "models", "best_house_price_model.pkl")

try:
    model, model_name, top_features = joblib.load(model_path)
except Exception as e:
    model, model_name, top_features = None, None, None
    print(f"Warning: Model could not be loaded from {model_path}. Error: {e}")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

def predict_price(request, predict_only=False):
    if model is None:
        raise ValueError("Model is not loaded.")
        
    semi = 1 if request.furnishingstatus == "semi-furnished" else 0
    unfurnished = 1 if request.furnishingstatus == "unfurnished" else 0

    input_dict = {
        "area": request.area,
        "bedrooms": request.bedrooms,
        "guestroom": 1 if request.guestroom == "Yes" else 0,
        "bathrooms": request.bathrooms,
        "mainroad": 1 if request.mainroad == "Yes" else 0,
        "prefarea": 1 if request.prefarea == "Yes" else 0,
        "stories": request.stories,
        "parking": request.parking,
        "basement": 1 if request.basement == "Yes" else 0,
        "airconditioning": 1 if request.airconditioning == "Yes" else 0,
        "furnishingstatus_semi-furnished": semi,
        "furnishingstatus_unfurnished": unfurnished
    }

    input_data = pd.DataFrame([input_dict])
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=top_features, fill_value=0)

    prediction = model.predict(input_data)[0]

    if predict_only:
        return {"prediction": float(prediction), "advisory": ""}
        
    prompt = f"""
    You are a professional and responsible real estate advisor.

    RULES:
    - Do NOT guarantee profits
    - Mention risks clearly
    - Keep advice realistic

    Property Details:
    Area: {request.area}
    Bedrooms: {request.bedrooms}
    Bathrooms: {request.bathrooms}
    Stories: {request.stories}
    Parking: {request.parking}
    Main Road: {request.mainroad}
    Preferred Area: {request.prefarea}
    Furnishing: {request.furnishingstatus}

    Predicted Price: ₹ {round(prediction, 2)}

    Generate:
    1. Property Summary
    2. Price Insight
    3. Market Trends
    4. Recommendation
    5. Risks
    """

    advisory = llm.invoke(prompt).content
    
    return {
        "prediction": float(prediction),
        "advisory": advisory
    }
