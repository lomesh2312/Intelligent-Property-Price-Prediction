from pydantic import BaseModel

class PredictionRequest(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    parking: int
    guestroom: str # "Yes" or "No"
    mainroad: str
    prefarea: str
    basement: str
    airconditioning: str
    furnishingstatus: str # "furnished", "semi-furnished", "unfurnished"

class PredictionResponse(BaseModel):
    prediction: float
    advisory: str

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
