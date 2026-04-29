# Smart Disease Diagnosis Chatbot

A full-stack AI-powered medical diagnosis assistant built with **FastAPI**, **React + Vite**, **TensorFlow/Keras**, and **SQLite**.

The system predicts possible diseases based on user symptoms and severity levels using a **BiLSTM-based deep learning model**. It also supports **history-aware prediction**, disease descriptions, precaution recommendations, and optional AI-generated explanations using Groq.

> This project is developed for educational and research purposes as part of a graduation project in Artificial Intelligence and Robotics.

---

## Overview

Smart Disease Diagnosis Chatbot is an intelligent medical support system that allows users to enter symptoms with severity levels and receive a structured prediction response.

The system provides:

- Predicted disease
- Confidence score
- Top possible conditions
- Disease description
- Suggested precautions
- Severity summary
- Current-session prediction
- History-aware prediction
- Optional AI-generated explanation

This system does **not replace doctors or professional medical diagnosis**. It is intended only as a supportive educational tool.

---

## Main Features

- Symptom-based disease prediction
- Severity-aware diagnosis
- History-aware inference using previous sessions
- FastAPI backend
- React + Vite frontend
- SQLite database for users, sessions, and prediction history
- BiLSTM deep learning model
- Disease description and precaution knowledge base
- Emergency-aware logic for critical symptom patterns
- Optional Groq AI explanation layer
- Swagger API documentation

---

## Tech Stack

### Backend

- Python 3.10
- FastAPI
- Uvicorn
- TensorFlow / Keras
- SQLite
- Pydantic
- python-dotenv
- Groq API optional integration

### Frontend

- React
- Vite
- Axios
- React Router DOM

### Machine Learning

- BiLSTM model
- Symptom encoding
- Severity encoding
- History-aware inference
- Confidence-based prediction logic

---

## Project Structure

```text
Backend/
├── data/
├── artifacts/
│   ├── db/
│   ├── figures/
│   └── models/
│
├── docs/
│   ├── core_workflow.txt
│   ├── project_steps.docx
│   └── other documentation files
│
├── Frontend/
│   └── disease-chatbot-frontend/
│       ├── public/
│       ├── src/
│       ├── package.json
│       ├── vite.config.js
│       └── README.md
│
├── src/
│   ├── api/
│   ├── ml/
│   ├── preprocessing/
│   ├── database/
│   └── services/
│
├── tests/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
How the System Works
The user enters symptoms and severity levels from the frontend.
The React frontend sends the request to the FastAPI backend.
The backend preprocesses the symptoms and severity values.
The trained BiLSTM model predicts the most likely disease.
The system calculates the confidence score and top predictions.
The knowledge base returns disease descriptions and precautions.
If previous sessions exist, the system can perform history-aware prediction.
The final structured response is displayed to the user.
Installation and Setup
1. Clone the repository
git clone https://github.com/alzoubitoqa/smart-disease-diagnosis-chatbot.git
cd smart-disease-diagnosis-chatbot
Backend Setup
2. Create and activate virtual environment
Windows PowerShell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
3. Install backend dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
4. Run the backend
python -m uvicorn src.api.main:app --reload

Backend URL:

http://127.0.0.1:8000

Swagger API documentation:

http://127.0.0.1:8000/docs
Frontend Setup
5. Move to the frontend folder
cd Frontend/disease-chatbot-frontend
6. Install frontend dependencies
npm install
7. Run the frontend
npm run dev

Frontend URL:

http://localhost:5173/
Environment Variables

Create a .env file in the backend root folder:

GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

Important:

Do not upload .env to GitHub.
If GROQ_API_KEY is empty, the backend can still run without AI explanation generation.
Dataset Files

Place the required CSV files inside:

data/raw/

Required files:

dataset.csv
symptom_Description.csv
symptom_precaution.csv
Symptom-severity.csv
Running Tests

From the backend root folder:

python tests/test_health.py
python tests/test_prediction.py
python tests/test_history.py
python tests/test_model_load.py
python tests/test_api.py
Prediction Modes
Current Session Prediction

Uses only the symptoms entered in the current session.

History-Aware Prediction

Uses the current symptoms together with selected symptoms from previous sessions to support more context-aware prediction.

Example Backend Endpoints
POST /api/auth/register
POST /api/auth/login
GET  /api/profile/{user_id}
GET  /api/symptoms
POST /api/predict
POST /api/predict/history-aware
GET  /api/history/{user_id}
Important Notes
Run backend commands from the root Backend folder.
Run frontend commands from Frontend/disease-chatbot-frontend.
The SQLite database is stored inside artifacts/db/.
Trained models are stored inside artifacts/models/.
Do not upload .env, .venv, or node_modules.
If the model files are too large for GitHub, use Git LFS or upload only the source code and documentation.
Disclaimer

This system is intended for educational and research purposes only.

It provides supportive predictions based on symptom patterns, severity values, and stored history. It does not replace medical professionals, clinical diagnosis, or emergency care.

Author

Developed by Toqa Al-Zoubi as part of a graduation project in Artificial Intelligence and Robotics.

Focus areas:

Medical diagnosis assistance
Deep learning using BiLSTM
History-aware inference
FastAPI backend development
React frontend integration
Applied AI in healthcare support systems
