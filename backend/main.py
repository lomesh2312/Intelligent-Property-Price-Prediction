from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
from schemas import PredictionRequest, PredictionResponse, ChatRequest, ChatResponse
from model import predict_price
from rag import rag

app = FastAPI(title="Intelligent Property Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        result = predict_price(request)
        return PredictionResponse(
            prediction=result["prediction"],
            advisory=result["advisory"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not rag:
        raise HTTPException(status_code=500, detail="RAG Engine unavailable")
    try:
        response = rag.query(request.query)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
