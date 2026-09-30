from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from core.ml.scorer import FraudScorer
from core.agents.investigator import FraudInvestigator
from typing import List, Tuple

app = FastAPI(title="SentinelAI API")

# Initialize Core Components
scorer = FraudScorer()
# In a real app, we would load a pre-trained model. 
# For the demo, we ensure it's trained on startup.
df = pd.read_csv('sentinel_ai/data/transactions.csv')
scorer.train(df)

investigator = FraudInvestigator()
graph = investigator.build_graph()

class TransactionRequest(BaseModel):
    transaction_id: str
    amount: float
    merchant_category: str
    location: str
    device_fingerprint: str
    velocity_1h: int
    velocity_24h: int
    avg_amount_30d: float
    distance_from_home_km: float
    foreign_transaction_flag: int

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/investigate")
async def investigate_transaction(req: TransactionRequest):
    try:
        # 1. Convert request to DataFrame for the ML pipeline
        data = {
            "transaction_id": [req.transaction_id],
            "user_id": ["UNKNOWN"], # Simplified for API
            "customer_name": ["Unknown"],
            "timestamp": ["2026-09-29T00:00:00"],
            "amount": [req.amount],
            "merchant_category": [req.merchant_category],
            "location": [req.location],
            "device_fingerprint": [req.device_fingerprint],
            "velocity_1h": [req.velocity_1h],
            "velocity_24h": [req.velocity_24h],
            "avg_amount_30d": [req.avg_amount_30d],
            "distance_from_home_km": [req.distance_from_home_km],
            "foreign_transaction_flag": [req.foreign_transaction_flag],
            "typology": ["normal"],
            "is_fraud": [0]
        }
        df_row = pd.DataFrame(data)
        
        # 2. Predict & Explain
        score = float(scorer.score(df_row)[0])
        evidence = scorer.explain(df_row)
        
        # 3. Trigger Agent if risk is high
        recommendation = "NONE"
        report = "Risk too low for full investigation."
        
        if score > 0.5: # Trigger threshold
            inputs = {
                "transaction_id": req.transaction_id,
                "risk_score": score,
                "shap_evidence": evidence,
                "retrieved_docs": [],
                "investigation_notes": []
            }
            result = graph.invoke(inputs)
            recommendation = result['recommendation']
            report = result['final_report']
            
        return {
            "transaction_id": req.transaction_id,
            "risk_score": score,
            "shap_drivers": evidence,
            "recommendation": recommendation,
            "forensic_report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
