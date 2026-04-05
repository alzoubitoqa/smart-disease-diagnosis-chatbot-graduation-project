from fastapi import APIRouter, HTTPException
from src.schemas.auth import RegisterRequest, LoginRequest, AuthResponse
from src.services.auth_service import register_user, login_user

router = APIRouter()


@router.post("/register", response_model=AuthResponse)
def register(payload: RegisterRequest):
    try:
        return register_user(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest):
    try:
        return login_user(payload)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))