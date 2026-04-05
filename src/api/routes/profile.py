from fastapi import APIRouter, HTTPException
from src.schemas.profile import ProfileRequest, ProfileResponse
from src.services.profile_service import get_profile, upsert_profile

router = APIRouter()


@router.get("/{user_id}", response_model=ProfileResponse)
def read_profile(user_id: int):
    try:
        return get_profile(user_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{user_id}", response_model=ProfileResponse)
def update_profile(user_id: int, payload: ProfileRequest):
    try:
        return upsert_profile(user_id, payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))