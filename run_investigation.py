import os
import pandas as pd
from core.ml.scorer import FraudScorer
from core.agents.investigator import FraudInvestigator

def simulate_fraud_case():
    # 1. Setup ML Scorer
    scorer = FraudScorer()
    # We'll assume the model is trained or we use a trained one from the previous step
    # For this demo, we load the data and train quickly
    df = pd.read_csv('sentinel_ai/data/transactions.csv')
    scorer.train(df)
    
    # 2. Pick a high-risk transaction (Fraud case)
    fraud_cases = df[df['is_fraud'] == 1]
    test_row = fraud_cases.head(1)
    txn_id = test_row['transaction_id'].values[0]
    
    # 3. Get ML Score and SHAP Evidence
    score = scorer.score(test_row)[0]
    evidence = scorer.explain(test_row)
    
    print(f"\n🚨 HIGH RISK ALERT: Transaction {txn_id}")
    print(f"Risk Score: {score:.4f}")
    print(f"Top Drivers: {evidence}\n")
    
    # 4. Trigger LangGraph Investigation
    investigator = FraudInvestigator()
    app = investigator.build_graph()
    
    inputs = {
        "transaction_id": txn_id,
        "risk_score": float(score),
        "shap_evidence": evidence,
        "retrieved_docs": [],
        "investigation_notes": []
    }
    
    print("🚀 Starting Agentic Investigation Workflow...")
    final_state = app.invoke(inputs)
    
    print("\n" + "="*50)
    print("📝 FINAL FORENSIC REPORT")
    print("="*50)
    print(f"Recommendation: {final_state['recommendation']}")
    print(f"\nAnalysis:\n{final_state['final_report']}")
    print("="*50)

if __name__ == "__main__":
    simulate_fraud_case()
