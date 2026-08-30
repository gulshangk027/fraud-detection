# MuleNet AI

![MuleNet AI Banner](https://img.shields.io/badge/MuleNet-AI-blueviolet?style=for-the-badge) ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi) ![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB) ![Vercel](https://img.shields.io/badge/vercel-%23000000.svg?style=for-the-badge&logo=vercel&logoColor=white)

MuleNet AI is a sophisticated, full-stack intelligence platform engineered for the financial sector to detect, analyze, and explain **mule accounts** and fraudulent financial networks. By leveraging state-of-the-art machine learning models and deep behavioral analytics, MuleNet AI provides risk analysts with an intuitive, explainable, and scalable dashboard to shut down financial cybercrime.

## 🌟 Key Features

*   **Drishthi (AI Explainability Engine):** Powered by SHAP (Shapley Additive exPlanations), Drishthi breaks down exactly *why* an account was flagged, showing the specific transaction behaviors (e.g., high UPI transaction velocity, abnormal cash withdrawals) driving the risk score.
*   **Calibrated XGBoost Classifier:** The core detection engine utilizes a Calibrated XGBoost architecture combined with an Isolation Forest for robust anomaly detection, providing highly accurate probability scores.
*   **Network Intelligence:** Analyzes the broader graph topology (Chain, Star, Ring formations) and calculates PageRank scores to uncover organized financial crime networks rather than just isolated accounts.
*   **Automated Investigation Reports:** Generates comprehensive, exportable markdown reports detailing executive summaries, risk interpretations, and recommended actions for compliance teams.
*   **Vercel Lite Mode:** A specialized deployment mode that automatically detects serverless environments (like Vercel). To comply with 250MB serverless function limits, the system dynamically swaps heavy ML dependencies for deterministic algorithmic mock engines, keeping the dashboard operational and beautiful.

## 🏗️ Architecture

MuleNet AI is built on a modern hybrid architecture:
*   **Frontend:** React / Vite (Single Page Application)
*   **Backend:** FastAPI (Python)
*   **Machine Learning:** XGBoost, Scikit-Learn, SHAP, Pandas, Numpy
*   **Deployment Infrastructure:** Configured for seamless deployment on Vercel using `vercel.json` rewrites and serverless Python functions (`api/index.py`).

## 🚀 Getting Started (Local Development)

To run the full MuleNet AI platform locally with complete machine learning capabilities:

### Prerequisites
*   Python 3.10+
*   Node.js 18+

### 1. Setup the Backend
Navigate to the backend directory and install the dependencies:
```bash
cd backend
python -m venv venv
venv\Scripts\activate   # (On Windows) or source venv/bin/activate (On Mac/Linux)
pip install -r requirements.txt
```

### 2. Start the Application
The FastAPI server is configured to serve both the API and the static React frontend from `frontend/dist`. 
```bash
python -m uvicorn app.main:app --port 8000
```
*Note: Make sure your frontend is built (`npm run build` in the frontend directory) before starting the server.*

Open your browser and navigate to `http://localhost:8000`.

## ☁️ Deployment (Vercel)

MuleNet AI is pre-configured for one-click deployment on Vercel. 
The repository includes a `vercel.json` configuration that seamlessly routes `/api/*` traffic to the FastAPI serverless function while serving the static frontend assets globally via Vercel's Edge Network.

When deployed to Vercel, the application automatically enters **Vercel Lite Mode** due to serverless constraints, ensuring 100% uptime and UI functionality using deterministic fallback algorithms without requiring heavy ML libraries.

## 🛡️ License

This project is intended for demonstration, portfolio, and educational purposes regarding AI in cybersecurity and financial fraud detection.
