from pydantic import BaseModel, Field
from typing import List, Optional


class SymptomInput(BaseModel):
    symptom: str = Field(..., min_length=1)
    severity: int = Field(..., ge=1, le=7)


class PredictionRequest(BaseModel):
    user_text: str
    symptoms: List[SymptomInput]


class HistoryAwarePredictionRequest(BaseModel):
    user_id: str
    user_text: str
    symptoms: Optional[List[SymptomInput]] = None
    default_severity: int = Field(1, ge=1, le=7)