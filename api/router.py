from fastapi import APIRouter, HTTPException
from src.predictor import Predictor
from api.schema import TextRequest, PredictionResposne, HealthResponse


router = APIRouter()

predictor: Predictor = None


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if predictor else "model not found",
        model_loaded=predictor is not None,
    )


@router.post("/predict", response_model=PredictionResposne)
def predict(request: TextRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="model not found")
    result = predictor.predict(request.text)
    return PredictionResposne(**result)
