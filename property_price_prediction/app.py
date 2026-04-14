import streamlit as st
import pandas as pd
import joblib
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

from rag_engine import RAGEngine


load_dotenv()

@st.cache_resource
def load_rag():
    return RAGEngine("property_price_prediction/data/real_estate_knowledge.txt")

rag = load_rag()


model_path = os.path.join(os.path.dirname(__file__), "model", "best_house_price_model.pkl")
model, model_name, top_features = joblib.load(model_path)


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "advisory" not in st.session_state:
    st.session_state.advisory = ""


st.title("🏠 Intelligent Property Price Prediction System")


st.header("Enter Property Details")

area = st.number_input("Area (sq ft)", min_value=1650, max_value=16200, value=None, placeholder="Enter area in sq ft greater than 1650")
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=None, placeholder="Enter number of bedrooms (1-6)")
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=4, value=None, placeholder="Enter number of bathrooms (1-4)")
stories = st.number_input("Stories", min_value=1, max_value=4, value=None, placeholder="Enter number of stories (1-4)")
parking = st.number_input("Parking Spaces", min_value=0, max_value=3, value=None,  placeholder="Enter number of parking spaces (0-3)")


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

    if area is None or bedrooms is None or bathrooms is None or stories is None or parking is None:
        st.error("⚠ Please fill all numeric input fields.")
        st.stop()

    if area <= 0 or bedrooms <= 0 or bathrooms <= 0:
        st.error("⚠ Please enter valid numeric values.")
        st.stop()


    semi = 1 if furnishingstatus == "semi-furnished" else 0
    unfurnished = 1 if furnishingstatus == "unfurnished" else 0

    input_dict = {
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
    }

    input_data = pd.DataFrame([input_dict])
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=top_features, fill_value=0)

    prediction = model.predict(input_data)[0]
    st.session_state.prediction = prediction

    st.success(f"💰 Predicted Price: ₹ {round(prediction, 2)}")
    st.info(f"📊 Model Used: {model_name}")


    prompt = f"""
    You are a professional and responsible real estate advisor.

    RULES:
    - Do NOT guarantee profits
    - Mention risks clearly
    - Keep advice realistic

    Property Details:
    Area: {area}
    Bedrooms: {bedrooms}
    Bathrooms: {bathrooms}
    Stories: {stories}
    Parking: {parking}
    Main Road: {mainroad}
    Preferred Area: {prefarea}
    Furnishing: {furnishingstatus}

    Predicted Price: ₹ {round(prediction, 2)}

    Generate:
    1. Property Summary
    2. Price Insight
    3. Market Trends
    4. Recommendation
    5. Risks
    """

    with st.spinner("Generating AI Advice..."):
        advisory = llm.invoke(prompt).content
        st.session_state.advisory = advisory

    st.subheader("🤖 AI Advisory Report")
    st.write(advisory)


st.divider()
st.header("📚 Legal Property Advisor (RAG Only)")

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask legal property questions")
    submit = st.form_submit_button("Ask")

if submit and user_input:
    rag_response = rag.query(user_input)

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Legal Advisor", rag_response))


st.subheader("💬 Chat History")

chat = st.session_state.chat_history

for i in range(len(chat) - 2, -1, -2):
    user_q = chat[i][1]
    bot_a = chat[i+1][1]

    st.markdown(f"**🧑 You:** {user_q}")
    st.markdown(f"**📚 Legal Advisor:** {bot_a}")
    st.divider()

if st.session_state.prediction is not None:

    full_report = f"""
🏠 REAL ESTATE REPORT

Predicted Price:
₹ {round(st.session_state.prediction, 2)}

-------------------------
AI ADVISORY
-------------------------
{st.session_state.advisory}

-------------------------
LEGAL Q&A (RAG)
-------------------------
"""

    for role, msg in st.session_state.chat_history:
        full_report += f"\n{role}: {msg}"

    st.download_button(
        label="📥 Download Full Report",
        data=full_report,
        file_name="real_estate_report.txt",
        mime="text/plain"
    )
