# Medical Diagnosis Assistant

An intelligent medical diagnosis chatbot with history-aware LSTM, FastAPI backend, Streamlit frontend, and optional Groq integration for AI explanations.

---

## 📁 Project Structure (after reorganization)
GP_2/
├── data/
│ └── raw/ # Raw CSV files
│ ├── dataset.csv
│ ├── symptom_Description.csv
│ ├── symptom_precaution.csv
│ └── Symptom-severity.csv
├── src/
│ ├── api/ # FastAPI endpoints
│ │ ├── api_main.py
│ │ └── groq_client.py
│ ├── database/ # Database helpers
│ │ ├── chatbot_db.py
│ │ └── review_history.py
│ ├── ml/ # Machine learning models
│ │ ├── predict_disease.py
│ │ ├── predict_disease_history.py
│ │ ├── train_bilstm.py
│ │ ├── train_baseline_rf.py
│ │ └── evaluate_group_kfold.py
│ └── preprocessing/ # Data preprocessing
│ ├── preprocess.py
│ ├── validate_data.py
│ ├── inspect_sample.py
│ └── build_knowledge_base.py
├── scripts/ # Standalone utility scripts
│ └── review_history.py
├── artifacts/ # Outputs (models, reports, db)
│ ├── models/
│ ├── reports/
│ ├── processed/
│ └── db/
├── .env # Environment variables
├── .env.example # Example environment variables
├── requirements.txt # Python dependencies
├── streamlit_app.py # Streamlit frontend
└── README.md # This file

text

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd GP_2
2. Create and activate virtual environment
bash
py -3.10 -m venv .venv
# On Windows:
.venv\Scripts\Activate.ps1
# On Linux/Mac:
source .venv/bin/activate
3. Upgrade pip and install dependencies
bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
4. Install additional packages (if not in requirements)
bash
pip install fastapi uvicorn streamlit requests python-dotenv groq
# Optional: TensorFlow (if not already installed)
pip install tensorflow==2.15.1
🧪 Preprocessing and Training
Make sure the raw CSV files are placed in data/raw/ before running these commands.

1. Preprocess the dataset
bash
python src/preprocessing/preprocess.py
2. Validate processed data
bash
python src/preprocessing/validate_data.py
3. Inspect processed samples (optional)
bash
python src/preprocessing/inspect_sample.py
4. Train BiLSTM model
bash
python src/ml/train_bilstm.py
5. Train RandomForest baseline
bash
python src/ml/train_baseline_rf.py
6. Run Group K-Fold evaluation
bash
python src/ml/evaluate_group_kfold.py
7. Build knowledge base (descriptions, precautions, severity)
bash
python src/preprocessing/build_knowledge_base.py
🤖 Prediction Scripts
Basic prediction (current session only)
bash
python src/ml/predict_disease.py
History-aware prediction (with database)
bash
python src/ml/predict_disease_history.py
Review saved user history
bash
python scripts/review_history.py
🚀 Running the API (Backend)
Start the FastAPI server:

bash
uvicorn src.api.api_main:app --reload
API will be available at http://127.0.0.1:8000

Interactive docs: http://127.0.0.1:8000/docs

Alternative docs: http://127.0.0.1:8000/redoc

🎨 Running the Streamlit App (Frontend)
In a separate terminal (with virtual environment activated):

bash
streamlit run streamlit_app.py
The app will open in your browser at http://localhost:8501.

🔐 Environment Variables
Create a .env file in the project root with the following variables:

env
API_BASE_URL=http://127.0.0.1:8000
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
To run without Groq (AI explanations disabled), leave GROQ_API_KEY empty.

📦 Full Run Order (Two Terminals)
Terminal 1 – Backend
bash
uvicorn src.api.api_main:app --reload
Terminal 2 – Frontend
bash
streamlit run streamlit_app.py
📝 Optional Logging
Save API logs to a file:

bash
uvicorn src.api.api_main:app --reload 2>&1 | Tee-Object -FilePath api_log.txt
Save Streamlit logs:

bash
streamlit run streamlit_app.py 2>&1 | Tee-Object -FilePath streamlit_log.txt
🧹 Notes
Ensure all raw CSV files are in data/raw/ before preprocessing.

The database files (app_users.db, medical_chatbot.db) will be created automatically in artifacts/db/.

If you encounter module import errors, make sure you are running commands from the project root (GP_2/).

For Windows PowerShell, use python instead of python3.