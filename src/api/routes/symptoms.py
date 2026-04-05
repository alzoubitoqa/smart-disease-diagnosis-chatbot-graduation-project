from fastapi import APIRouter
from src.services.symptom_service import get_all_symptoms

router = APIRouter()


@router.get("/")
def list_symptoms():
    return get_all_symptoms()