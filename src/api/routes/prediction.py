from fastapi import APIRouter
from src.schemas.prediction import PredictionRequest, HistoryAwarePredictionRequest
from src.services.prediction_service import run_prediction, run_history_aware_prediction

router = APIRouter()


@router.post("/predict")
def predict_disease(payload: PredictionRequest):
    return run_prediction(payload)


@router.post("/predict/history-aware")
def predict_disease_history_aware(payload: HistoryAwarePredictionRequest):
    return run_history_aware_prediction(payload)