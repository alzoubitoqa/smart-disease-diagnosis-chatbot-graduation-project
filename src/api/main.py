from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.prediction import router as prediction_router
from src.api.routes.auth import router as auth_router
from src.api.routes.profile import router as profile_router
from src.api.routes.symptoms import router as symptoms_router
from src.api.routes.history import router as history_router
from src.core.database import init_db

app = FastAPI(title="Disease Prediction Chatbot API")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    init_db()


app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(profile_router, prefix="/api/profile", tags=["Profile"])
app.include_router(symptoms_router, prefix="/api/symptoms", tags=["Symptoms"])
app.include_router(history_router, prefix="/api/history", tags=["History"])
app.include_router(prediction_router, prefix="/api", tags=["Prediction"])


@app.get("/")
def root():
    return {"message": "API is running"}