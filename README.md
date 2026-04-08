أكيد — بما إن الريبو عندك **فاضي** على GitHub واسمُه:

`Disease-chatbot-BACKEND-Final-edit`

فالأفضل تحطي **README احترافي جدًا وجاهز للرفع** من أول مرة.
هذا إصدار مرتب جدًا ومناسب لمشروعك الحالي مع **FastAPI + React Vite + BiLSTM + History-Aware Prediction**:

````md
# Medical Diagnosis Assistant Backend

A history-aware medical diagnosis backend built with **FastAPI**, **TensorFlow/Keras**, and **SQLite**.  
It predicts possible diseases from user symptoms and severity levels using a **BiLSTM-based deep learning model**, supports **history-aware inference**, and can optionally generate AI explanations using **Groq**.

> This backend is designed to work with a **React + Vite frontend**.

---

## Overview

This project is part of an intelligent medical diagnosis assistant that helps users enter symptoms and severity levels, then returns:

- possible disease prediction
- confidence score
- top possible conditions
- disease description
- precautions
- severity summary
- optional history-aware prediction
- optional AI-generated explanation

The system is designed for **educational and research purposes only** and does **not replace professional medical diagnosis**.

---

## Main Features

- Symptom-based disease prediction
- Severity-aware diagnosis
- History-aware prediction using previous sessions
- FastAPI REST API backend
- SQLite database for storing users, sessions, and predictions
- Knowledge base integration for descriptions and precautions
- Emergency-aware logic for critical symptom patterns
- React + Vite frontend integration
- Optional Groq explanation layer

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

### Machine Learning
- BiLSTM model
- Symptom embedding
- Severity embedding
- History-aware inference
- Confidence-based decision logic

### Frontend Integration
- React
- Vite
- REST API communication

---

## Project Structure

```text
Backend/
├── data/
│   └── raw/
│       ├── dataset.csv
│       ├── symptom_Description.csv
│       ├── symptom_precaution.csv
│       └── Symptom-severity.csv
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py
│   │   ├── main.py
│   │   └── routes/
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── database.py
│   │   ├── groq_client.py
│   │   ├── logger.py
│   │   ├── security.py
│   │   └── utils.py
│   │
│   ├── knowledge_base/
│   ├── ml/
│   │   ├── evaluate.py
│   │   ├── history_inference.py
│   │   ├── inference.py
│   │   ├── loaders.py
│   │   └── train_bilstm.py
│   │
│   ├── models/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── encoders.py
│   │   ├── preprocessor.py
│   │   ├── sequence_builder.py
│   │   └── text_cleaning.py
│   │
│   ├── repositories/
│   ├── schemas/
│   ├── services/
│   │   ├── auth_service.py
│   │   ├── history_inference_service.py
│   │   ├── history_service.py
│   │   ├── prediction_rules_service.py
│   │   ├── prediction_service.py
│   │   ├── profile_service.py
│   │   ├── recommendation_service.py
│   │   ├── symptom_parser_service.py
│   │   └── symptom_service.py
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── generate_figures.py
│       └── generate_roc_curves.py
│
├── artifacts/
│   ├── db/
│   ├── models/
│   ├── processed/
│   └── reports/
│
├── tests/
│   ├── comprehensive_testing.py
│   ├── comprehensive_validation.py
│   ├── test_api.py
│   ├── test_health.py
│   ├── test_history.py
│   ├── test_model_load.py
│   ├── test_overfitting.py
│   ├── test_prediction.py
│   └── test_underfitting.py
│
├── .env
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
````

---

## How the System Works

1. The user enters symptoms and severity values from the frontend.
2. The frontend sends a request to the FastAPI backend.
3. The backend preprocesses the input symptoms.
4. The trained BiLSTM model predicts the most likely disease.
5. The system calculates:

   * confidence score
   * top-k predictions
   * severity summary
6. The system retrieves:

   * disease description
   * precautions
7. If previous history exists, the backend may perform a history-aware prediction.
8. If enabled, Groq generates a natural-language explanation.
9. The final structured response is sent back to the frontend.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/alzoubitoqa/Disease-chatbot-BACKEND-Final-edit.git
cd Disease-chatbot-BACKEND-Final-edit
```

### 2. Create and activate a virtual environment

#### Windows PowerShell

```bash
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Optional manual installation

If some packages are missing from `requirements.txt`, install them manually:

```bash
pip install fastapi uvicorn python-dotenv requests groq
pip install tensorflow==2.15.1
```

---

## Dataset Preparation

Place the following raw CSV files inside:

```text
data/raw/
```

Required files:

* `dataset.csv`
* `symptom_Description.csv`
* `symptom_precaution.csv`
* `Symptom-severity.csv`

---

## Preprocessing

Run preprocessing before training or inference if processed artifacts do not exist.

### Preprocess the dataset

```bash
python src/preprocessing/preprocessor.py
```

### Generate figures

```bash
python src/visualization/generate_figures.py
python src/visualization/generate_roc_curves.py
```

---

## Model Training

### Train the BiLSTM model

```bash
python src/ml/train_bilstm.py
```

### Evaluate the model

```bash
python src/ml/evaluate.py
```

---

## Running Tests

### API and functionality tests

```bash
python tests/test_api.py
python tests/test_health.py
python tests/test_prediction.py
python tests/test_history.py
python tests/test_model_load.py
```

### Validation and model behavior tests

```bash
python tests/comprehensive_testing.py
python tests/comprehensive_validation.py
python tests/test_overfitting.py
python tests/test_underfitting.py
```

---

## Running the Backend

Start the FastAPI server:

```bash
python -m uvicorn src.api.main:app --reload
```

Backend will run at:

```text
http://127.0.0.1:8000
```

### API Documentation

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

---

## Frontend Integration

This backend is connected to a **React + Vite frontend**.

Example frontend API base URL:

```js
const API_BASE_URL = "http://127.0.0.1:8000";
```

Example using Axios:

```js
import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:8000",
});
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
API_BASE_URL=http://127.0.0.1:8000
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

### Notes

* If `GROQ_API_KEY` is empty, the backend should still work without AI explanation generation.
* Do not upload your `.env` file publicly.

---

## Prediction Modes

### 1. Current Session Prediction

Prediction based only on the symptoms entered in the current session.

### 2. History-Aware Prediction

Prediction based on:

* current session symptoms
* selected useful symptoms from previous sessions

If the current prediction is already clear or medically important, the system may keep the current prediction unchanged for safety and stability.

---

## Core Modules

### Prediction Service

Handles:

* current prediction
* confidence handling
* top-k outputs
* severity summary

### History Service

Handles:

* session storage
* previous symptoms retrieval
* history merging
* history-aware prediction support

### Knowledge Base

Provides:

* disease descriptions
* precautions
* symptom severity logic

### Emergency Rule Layer

Adds extra safety logic for critical symptom patterns such as:

* chest pain
* breathlessness
* severe dizziness
* neurological warning signs

---

## Example Output

The backend can return:

* predicted disease
* confidence percentage
* top 3 possible conditions
* disease description
* precautions
* severity summary
* current-session result
* history-aware result
* emergency flag
* AI explanation

---

## Important Notes

* Run all commands from the project root folder.
* The SQLite database is created automatically inside `artifacts/db/`.
* Trained models are saved inside `artifacts/models/`.
* Processed artifacts are saved inside `artifacts/processed/`.
* If import errors appear, make sure:

  * the virtual environment is activated
  * all dependencies are installed
  * you are running commands from the correct root folder

---

## Disclaimer

This system is intended for **educational and research purposes only**.
It provides supportive predictions based on symptom patterns and stored history, but it does **not replace medical professionals, clinical diagnosis, or emergency care**.

---

## Author

Developed as part of a graduation project in **Artificial Intelligence and Robotics**, focusing on:

* medical diagnosis assistance
* deep learning with BiLSTM
* history-aware inference
* backend API development
* React frontend integration
* applied AI in healthcare support systems

