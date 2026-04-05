from pydantic import BaseModel
from typing import Optional


class ProfileRequest(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    medical_history: Optional[str] = None


class ProfileResponse(BaseModel):
    user_id: int
    age: Optional[int] = None
    gender: Optional[str] = None
    medical_history: Optional[str] = None