import pandas as pd
from core.ml.scorer import FraudScorer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np

def run_training():
    print("🚀 Loading dataset...")
    df = pd.read_csv('sentinel_ai/data/transactions.csv')
    
    scorer = FraudScorer()
    
    print("🛠️ Training ensemble model (XGBoost + Isolation Forest)...")
    scorer.train(df)
    
    # Evaluation
    scores = scorer.score(df)
    labels = df['is_fraud']
    
    roc_auc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    
    print(f"\n✅ Baseline Evaluation Results:")
    print(f"----------------------------")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    
    # Test SHAP explanation on a known fraud case
    fraud_cases = df[df['is_fraud'] == 1]
    if not fraud_cases.empty:
        test_row = fraud_cases.head(1)
        explanation = scorer.explain(test_row)
        print(f"\n🔍 Testing SHAP Explanation for a Fraud Case:")
        for feature, val in explanation:
            print(f"- {feature}: {val:.4f}")

if __name__ == "__main__":
    run_training()
