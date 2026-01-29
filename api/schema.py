from pydantic import BaseModel, Field
from typing import Dict


class TextRequest(BaseModel):
    text: str = Field(..., min_length=5)


class PredictionResposne(BaseModel):
    text: str
    predicted_label: str
    predicted_id: int
    confidence: float
    probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
