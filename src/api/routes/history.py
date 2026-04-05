from fastapi import APIRouter, HTTPException
from src.services.history_service import get_user_history

router = APIRouter()


@router.get("/{user_id}")
def read_user_history(user_id: int):
    try:
        return get_user_history(user_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))