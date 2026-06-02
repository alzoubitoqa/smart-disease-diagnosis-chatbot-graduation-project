# A Smart Chatbot System for Diagnosing Diseases

**Author:** Toqa Mahmoud Tawfiq Al-Zoubi

**Supervisor:** Prof. Abdelwadood Mesleh

**Program:** Artificial Intelligence and Robotics

**Version:** v1.0 Final Release

**Copyright © 2026 Toqa Mahmoud Tawfiq Al-Zoubi**

---

An AI-powered medical diagnosis support system designed to predict possible diseases based on user-reported symptoms and symptom severity levels using deep learning techniques.

The system combines a BiLSTM-based prediction model, symptom severity analysis, disease knowledge-base integration, user session history, and a chatbot interface to provide intelligent healthcare support.

---

# Project Information

**Project Title:**
A Smart Chatbot System for Diagnosing Diseases

**Field:**
Artificial Intelligence in Healthcare

**Academic Program:**
Artificial Intelligence and Robotics

**Supervisor:**
Prof. Abdelwadood Mesleh

---

# Author

## System Design and Implementation

**Toqa Mahmoud Tawfiq Al-Zoubi**

The implementation presented in this repository includes:

- System architecture design
- Backend development
- Frontend development
- Database integration
- Deep learning model implementation
- Data preprocessing pipeline
- Disease prediction engine
- Knowledge-base integration
- User authentication system
- Session-history management
- API development and testing
- Deployment preparation
- Technical documentation

---

# Project Overview

The Smart Disease Diagnosis Chatbot is an intelligent healthcare support system that assists users in identifying possible diseases based on their symptoms and severity levels.

The system utilizes a Bidirectional Long Short-Term Memory (BiLSTM) neural network trained on disease-symptom datasets to predict potential medical conditions.

The chatbot provides:

- Predicted disease
- Confidence score
- Ranked disease candidates
- Disease description
- Recommended precautions
- Symptom severity analysis
- Session-aware support
- History-aware prediction
- Safety-oriented medical guidance

This system is intended as a decision-support tool and does not replace professional medical diagnosis.

---

# Main Features

### User Features

- User registration and login
- Secure authentication
- User profile management
- Symptom selection interface
- Symptom severity selection
- Disease prediction
- Disease explanation
- Precaution recommendations
- Prediction history tracking

### AI Features

- BiLSTM-based disease prediction
- Symptom-severity encoding
- Confidence-based classification
- Ranked prediction generation
- History-aware inference
- Emergency symptom detection
- Low-confidence warning system

### System Features

- FastAPI backend
- React frontend
- SQLite database
- REST API architecture
- Knowledge-base integration
- Modular software architecture
- Swagger API documentation

---

# Technology Stack

## Backend

- Python 3.10
- FastAPI
- Uvicorn
- TensorFlow
- Keras
- SQLite
- Pydantic
- JWT Authentication
- Bcrypt
- Python-dotenv

## Frontend

- React
- Vite
- Axios
- React Router DOM

## Artificial Intelligence

- Deep Learning
- Bidirectional LSTM (BiLSTM)
- Symptom Encoding
- Severity Encoding
- Sequence Classification
- Confidence Scoring

---

# Project Structure

```text
Backend
│
├── .venv/
├── artifacts/
├── data/
├── DiagnoSeq/
│
├── docs/
│   ├── core_workflow.txt
│   ├── delet_command.txt
│   ├── final result.docx
│   ├── initial_plan.docx
│   └── project_steps.docx
│
├── Frontend/
│
├── src/
│   ├── api/
│   ├── artifacts/
│   ├── chatbot/
│   ├── core/
│   ├── knowledge_base/
│   ├── ml/
│   ├── models/
│   ├── preprocessing/
│   ├── repositories/
│   ├── schemas/
│   ├── services/
│   ├── visualization/
│   └── __init__.py
│
├── tests/
│
├── .env
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
```

---

# System Workflow

### Step 1

The user enters symptoms through the chatbot interface.

### Step 2

The user assigns severity levels to each selected symptom.

### Step 3

The frontend sends the data to the FastAPI backend.

### Step 4

The preprocessing module transforms symptoms and severity values into encoded sequences.

### Step 5

The BiLSTM model processes the encoded symptom sequence.

### Step 6

The prediction engine generates disease probabilities.

### Step 7

The system calculates:

- Predicted disease
- Confidence score
- Alternative predictions

### Step 8

The knowledge base retrieves:

- Disease description
- Recommended precautions

### Step 9

The chatbot presents the final response to the user.

---

# Core Components

## Authentication Module

Responsible for:

- Registration
- Login
- Password hashing
- JWT token generation

---

## Profile Module

Responsible for:

- User information
- Medical profile data
- Session ownership

---

## Prediction Module

Responsible for:

- Symptom processing
- Severity processing
- Disease prediction
- Confidence calculation

---

## History Module

Responsible for:

- Session storage
- Previous predictions
- History-aware prediction support

---

## Knowledge Base

Responsible for:

- Disease descriptions
- Disease precautions
- Safety-oriented guidance

---

# Dataset

The model was trained using disease-symptom datasets containing:

- Disease names
- Symptom combinations
- Symptom severity information
- Disease descriptions
- Precaution recommendations

The final system supports prediction across multiple disease categories represented within the training dataset.

---

# Model Architecture

The prediction engine is based on an enhanced BiLSTM architecture that processes:

- Symptom sequences
- Severity sequences

Model capabilities include:

- Long-term dependency learning
- Bidirectional context understanding
- Confidence-based prediction
- Medical sequence classification

---

# Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Start Backend

```bash
python run.py
```

or

```bash
uvicorn src.api.main:app --reload
```

---

# API Documentation

After running the backend:

```text
http://localhost:8000/docs
```

Swagger documentation provides full access to all available API endpoints.

---

# Important Notice

This project is intended for:

- Educational purposes
- Research purposes
- Artificial Intelligence experimentation

The generated predictions should not be considered medical diagnoses and must not replace professional healthcare consultation.

---

# Copyright

© 2026 Toqa Mahmoud Tawfiq Al-Zoubi

All source-code implementation, system integration, architecture development, machine-learning pipeline implementation, backend services, frontend integration, and technical documentation contained in this repository are attributed to the author listed above.

## License

Copyright © 2026 Toqa Mahmoud Tawfiq Al-Zoubi

All Rights Reserved.

Unauthorized copying, redistribution, modification, publication, reproduction, or use of this project, in whole or in part, without prior written permission from the author is prohibited.

This repository is maintained as the official implementation archive of the graduation project:

"A Smart Chatbot System for Diagnosing Diseases"

Version: v1.0 Final Release
