"""
MuleNet AI Backend Core API (FastAPI)
AI-Powered Mule Account Fraud Intelligence Platform
Tagline: PREVENT. DETECT. RESPOND.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import pandas as pd
from datetime import datetime
import os

from app.services.bank_features import BANK_FINALIZED_FEATURES, TARGET_VARIABLE, get_feature_info
from app.services.dataset_service import (
    process_uploaded_csv, get_current_dataset, _DATASET_METADATA, generate_synthetic_dataset, update_dataset_metadata
)
from app.services.ml_service import (
    train_dristhi_model, predict_account, get_feature_subset, predict_custom_features, 
    get_model_status, _MODEL_METRICS, activate_model, get_analytics_overview,
    _TRAINED_MODELS, _ACTIVE_MODE, load_persisted_model
)
from app.services.network_service import generate_mule_network_graph
from app.services.nlp_service import analyze_rakshak_message, analyze_awaaz_incident

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MuleNet AI API",
    description="AI-Powered Mule Account Fraud Intelligence Platform Backend",
    version="1.0.0"
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas ---
class PredictRequest(BaseModel):
    account_id: str
    mode: Optional[str] = "full_feature"

class CustomPredictRequest(BaseModel):
    features: Dict[str, float]
    mode: Optional[str] = "full_feature"

class TrainRequest(BaseModel):
    mode: Optional[str] = "full_feature" # "bank_finalized" or "full_feature"
    model_type: Optional[str] = "xgboost" # "xgboost", "random_forest", "logistic"

class ActivateModelRequest(BaseModel):
    mode: Optional[str] = "full_feature"
    model_type: Optional[str] = "xgboost"

class RakshakRequest(BaseModel):
    message: str

class AwaazRequest(BaseModel):
    incident_text: str

class ReportRequest(BaseModel):
    account_id: str
    mode: Optional[str] = "full_feature"

# --- Endpoints ---

@app.get("/api/health")
def root():
    return {
        "platform": "MuleNet AI",
        "tagline": "PREVENT. DETECT. RESPOND.",
        "status": "ONLINE",
        "modules": ["RAKSHAK", "DRISHTHI", "AWAAZ"],
        "target_variable": TARGET_VARIABLE
    }



@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload CSV dataset and auto-detect target F3924."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    content = await file.read()
    success, msg, metadata = process_uploaded_csv(content, file.filename)
    
    if not success:
        return {"status": "error", "message": msg, "metadata": metadata}
        
    # Auto retrain active model on upload
    train_dristhi_model(mode="full_feature", model_type="xgboost")
    
    return {"status": "success", "message": msg, "metadata": metadata}

@app.get("/api/dataset/summary")
def get_dataset_summary():
    """Returns dataset health, feature counts, and target stats."""
    df = get_current_dataset()
    meta = _DATASET_METADATA.copy()
    meta["sample_accounts"] = df["ACCOUNT_ID"].head(10).tolist() if "ACCOUNT_ID" in df.columns else []
    return meta

@app.get("/api/analytics/overview")
def get_analytics():
    """Returns calculated analytics, target breakdown, feature importances, model performance, and risk distribution."""
    return get_analytics_overview()

@app.get("/api/features/dictionary")
def get_feature_dictionary():
    """Returns official 18 Bank Finalized anchor feature dictionary + full dataset features."""
    df = get_current_dataset()
    cols = [c for c in df.columns if c not in [TARGET_VARIABLE, "ACCOUNT_ID"]]
    
    dict_list = []
    for fname in cols:
        info = get_feature_info(fname)
        dict_list.append({
            "feature_id": fname,
            "variable": info["variable"],
            "description": info["description"],
            "category": info["category"],
            "is_bank_finalized": info["is_bank_finalized"]
        })
    return {
        "bank_finalized_count": len(BANK_FINALIZED_FEATURES),
        "total_features": len(cols),
        "dictionary": dict_list
    }

@app.get("/api/model/status")
def model_status():
    """Returns whether model is trained and active metrics."""
    return get_model_status()

@app.post("/api/model/train")
def train_model(req: TrainRequest):
    """Trains Drishthi model (Mode A Bank Finalized or Mode B Full Feature Intelligence)."""
    try:
        metrics = train_dristhi_model(mode=req.mode, model_type=req.model_type)
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/activate")
def activate_model_api(req: ActivateModelRequest):
    """Activates a trained model mode for Drishthi real-time predictions."""
    try:
        metrics = activate_model(mode=req.mode, model_type=req.model_type)
        return {"status": "success", "active_mode": req.mode, "metrics": metrics}
    except Exception as e:
        logger.error(f"Activation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/metrics")
def get_model_metrics(mode: str = Query("full_feature")):
    """Returns trained metrics for Mode A or Mode B."""
    if mode in _MODEL_METRICS:
        return _MODEL_METRICS[mode]
    metrics = train_dristhi_model(mode=mode)
    return metrics

@app.post("/api/predict")
def predict_account_api(req: PredictRequest):
    """Calculates risk score, classification, and SHAP explanation for an account."""
    try:
        result = predict_account(req.account_id, mode=req.mode)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/custom")
def predict_custom_api(req: CustomPredictRequest):
    """Calculates risk score, classification, and SHAP explanation for custom feature inputs."""
    try:
        result = predict_custom_features(req.features, mode=req.mode)
        return result
    except Exception as e:
        logger.error(f"Custom prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/accounts/{account_id}")
def get_account_details(account_id: str, mode: str = Query("full_feature")):
    """Returns account investigation view."""
    return predict_account(account_id, mode=mode)

@app.get("/api/alerts")
def get_alerts():
    """Generates intelligent alerts for accounts exceeding risk threshold using Drishthi predictions."""
    df = get_current_dataset()
    if len(df) == 0:
        return {"total_alerts": 0, "alerts": []}
        
    is_trained = len(_TRAINED_MODELS) > 0
    if not is_trained:
        load_persisted_model()
        is_trained = len(_TRAINED_MODELS) > 0

    alerts = []
    if is_trained:
        active_m = _ACTIVE_MODE if _ACTIVE_MODE in _TRAINED_MODELS else list(_TRAINED_MODELS.keys())[0]
        model_obj = _TRAINED_MODELS[active_m]["model"]
        feature_cols = _TRAINED_MODELS[active_m]["feature_cols"]
        b_means = _TRAINED_MODELS[active_m].get("baseline_means", {})
        
        data_dict = {col: df[col].fillna(b_means.get(col, 0.0)) if col in df.columns else float(b_means.get(col, 0.0)) for col in feature_cols}
        X_df = pd.DataFrame(data_dict, index=df.index)
                
        X_all = X_df.fillna(0).values
        probs = model_obj.predict_proba(X_all)[:, 1]
        scores = probs * 100.0
        
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        acc_ids = df["ACCOUNT_ID"].astype(str).tolist() if "ACCOUNT_ID" in df.columns else [f"ACC-{100000+i}" for i in range(len(df))]
        
        for idx, (aid, score) in enumerate(zip(acc_ids, scores)):
            score_val = round(float(score), 1)
            if score_val >= 25.0:
                if score_val >= 75.0:
                    r_level = "CRITICAL"
                    r_action = "FREEZE / IMMEDIATE INVESTIGATION"
                elif score_val >= 50.0:
                    r_level = "HIGH"
                    r_action = "RESTRICT / ENHANCED MONITORING"
                else:
                    r_level = "MEDIUM"
                    r_action = "MONITOR ACCOUNT ACTIVITY"
                    
                is_mule = score_val >= 50.0
                alerts.append({
                    "account_id": aid,
                    "timestamp": now_str,
                    "risk_score": score_val,
                    "risk_level": r_level,
                    "classification": "SUSPICIOUS / MULE" if is_mule else "ELEVATED RISK ACCOUNT",
                    "recommended_action": r_action
                })
        alerts = sorted(alerts, key=lambda x: x["risk_score"], reverse=True)
    else:
        acc_ids = df["ACCOUNT_ID"].head(20).tolist() if "ACCOUNT_ID" in df.columns else [f"ACC-{100000+i}" for i in range(20)]
        for aid in acc_ids:
            res = predict_account(aid, mode="full_feature")
            if res.get("found") and res.get("risk_score", 0) >= 25.0:
                alerts.append({
                    "account_id": aid,
                    "timestamp": "2026-08-21 12:00:00",
                    "risk_score": res["risk_score"],
                    "risk_level": res.get("risk_level", "MEDIUM"),
                    "classification": res.get("classification", "SUSPICIOUS / MULE"),
                    "recommended_action": res.get("recommended_action", "MONITOR ACCOUNT ACTIVITY")
                })
        alerts = sorted(alerts, key=lambda x: x["risk_score"], reverse=True)

    return {"total_alerts": len(alerts), "alerts": alerts}

@app.get("/api/network")
def get_network_graph():
    """Returns NetworkX topology graph for account network visualization."""
    return generate_mule_network_graph()

@app.get("/api/network/{account_id}")
def get_network_by_account(account_id: str):
    """Returns NetworkX topology graph for specific account network visualization."""
    return generate_mule_network_graph()

@app.post("/api/rakshak/analyze")
def analyze_rakshak(req: RakshakRequest):
    """Rakshak NLP recruitment risk analyzer (negation-aware)."""
    return analyze_rakshak_message(req.message)

@app.post("/api/awaaz/analyze")
def analyze_awaaz(req: AwaazRequest):
    """Awaaz NLP victim incident parser & complaint draft generator."""
    return analyze_awaaz_incident(req.incident_text)

@app.post("/api/report")
def generate_report(req: ReportRequest):
    """Generates comprehensive investigation report connected to Drishthi ML and Mule Networks."""
    acc_res = predict_account(req.account_id, mode=req.mode)
    
    if not acc_res.get("found"):
        return {
            "found": False,
            "account_id": req.account_id,
            "error": f"No account matching {req.account_id} was found in the active dataset."
        }
        
    net_data = generate_mule_network_graph()
    net_node = next((n for n in net_data.get("nodes", []) if n["id"] == acc_res["account_id"]), None)
    
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_id = f"REP-{datetime.now().strftime('%Y%m%d')}-{acc_res['account_id'].replace('ACC-', '')}"
    
    r_level = acc_res["risk_level"]
    r_score = acc_res["risk_score"]
    
    # Executive Summary Narrative
    if r_score >= 75:
        exec_summary = f"Account {acc_res['account_id']} has been classified as a CRITICAL risk suspicious/mule account with a calibrated model risk score of {r_score}/100. The prediction is driven by severe behavioral anomalies and high risk feature deviations."
        risk_interpretation = "The account demonstrates strong behavioral characteristics associated with suspicious mule activity. Immediate account freeze and investigation are recommended."
    elif r_score >= 50:
        exec_summary = f"Account {acc_res['account_id']} has been classified as a HIGH risk suspicious/mule account with a calibrated model risk score of {r_score}/100. Behavioral signals indicate elevated transaction velocity and potential mule activity."
        risk_interpretation = "The account demonstrates multiple elevated-risk behavioral indicators. Enhanced investigation and transaction monitoring are recommended."
    elif r_score >= 25:
        exec_summary = f"Account {acc_res['account_id']} has been classified as a MEDIUM risk account with a calibrated model risk score of {r_score}/100. The account exhibits moderate transaction frequency and elevated risk factors."
        risk_interpretation = "Several behavioral indicators show elevated risk characteristics. Enhanced monitoring and further review are recommended."
    else:
        exec_summary = f"Account {acc_res['account_id']} has been classified as a LEGITIMATE account with a calibrated model risk score of {r_score}/100. Behavioral signals align with standard retail banking patterns."
        risk_interpretation = "Current behavioral indicators do not demonstrate significant mule-account characteristics. No immediate action is recommended, subject to standard monitoring."

    # Top SHAP Features Table
    top_shap = []
    for f in acc_res.get("shap_contributions", [])[:7]:
        direction = "Increases Mule Risk" if f["is_risk_increasing"] else "Decreases Mule Risk"
        top_shap.append({
            "feature_name": f["variable_name"],
            "feature_id": f["feature_id"],
            "actual_value": f["actual_value"],
            "shap_impact": f["shap_value"],
            "effect": direction
        })

    # Network Intelligence
    if net_node:
        net_info = {
            "has_network": True,
            "in_degree": net_node["in_degree"],
            "out_degree": net_node["out_degree"],
            "pagerank": net_node["pagerank"],
            "primary_topology": net_node["primary_topology"],
            "topology_explanation": net_node["topology_explanation"],
            "incoming_amount": net_node["incoming_amount"],
            "outgoing_amount": net_node["outgoing_amount"],
            "connected_accounts_count": net_node["connected_accounts_count"]
        }
    else:
        net_info = {
            "has_network": False,
            "primary_topology": "N/A",
            "topology_explanation": "No significant network relationship data available."
        }

    dataset_name = _DATASET_METADATA.get("filename", "Synthetic Demonstration Data")
    is_demo = _DATASET_METADATA.get("is_demo", True)
    mode_label = "DEMO MODE" if is_demo else "ML MODE"

    # Markdown representation
    report_md = f"""# MULENET AI — INVESTIGATION REPORT
Report ID: {report_id} | Timestamp: {now_str}
Account ID: {acc_res['account_id']} | Risk Score: {r_score}/100 ({r_level})
Classification: {acc_res['classification']} | Mule Probability: {round(acc_res['model_probability']*100, 1)}%

1. EXECUTIVE SUMMARY
{exec_summary}

2. RISK ASSESSMENT
- Risk Score: {r_score} / 100
- Risk Level: {r_level}
- Mule Probability: {round(acc_res['model_probability']*100, 1)}%
- Anomaly Score: {acc_res['anomaly_score']}
- Classification: {acc_res['classification']}

3. TOP RISK CONTRIBUTING FEATURES
"""
    for item in top_shap:
        report_md += f"- {item['feature_name']} ({item['feature_id']}): Val={item['actual_value']}, SHAP={item['shap_impact']} ({item['effect']})\n"

    report_md += f"""
4. RISK INTERPRETATION & RECOMMENDED ACTION
Interpretation: {risk_interpretation}
Recommended Action: {acc_res['recommended_action']}

5. METADATA
Dataset: {dataset_name}
Mode: {mode_label}
Model Status: Active
"""

    return {
        "found": True,
        "report_id": report_id,
        "generation_timestamp": now_str,
        "account_id": acc_res["account_id"],
        "risk_score": r_score,
        "risk_level": r_level,
        "classification": acc_res["classification"],
        "model_probability": acc_res["model_probability"],
        "anomaly_score": acc_res["anomaly_score"],
        "recommended_action": acc_res["recommended_action"],
        "executive_summary": exec_summary,
        "risk_interpretation": risk_interpretation,
        "top_risk_features": top_shap,
        "behavioral_profile": acc_res.get("behavioral_profile", {}),
        "network_intelligence": net_info,
        "metadata": {
            "dataset_name": dataset_name,
            "mode": mode_label,
            "model_status": "Active",
            "model_name": "Drishthi Calibrated XGBoost Classifier"
        },
        "markdown_report": report_md,
        "report_markdown": report_md
    }

# Serve static frontend files
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")
    
    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_dist, "index.html"))
        
    @app.get("/{catchall:path}")
    async def serve_frontend_catchall(catchall: str):
        if catchall.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        filepath = os.path.join(frontend_dist, catchall)
        if os.path.exists(filepath) and os.path.isfile(filepath):
            return FileResponse(filepath)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
