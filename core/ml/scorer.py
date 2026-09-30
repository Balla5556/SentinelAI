import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest
import shap
import pickle
import os

class FraudScorer:
    def __init__(self, model_path='sentinel_ai/models/fraud_model.pkl'):
        self.model_path = model_path
        self.xgb_model = None
        self.iso_forest = None
        self.feature_pipeline = None

    def train(self, df, target_col='is_fraud'):
        from .features import FeaturePipeline
        
        # 1. Feature Engineering
        self.feature_pipeline = FeaturePipeline()
        self.feature_pipeline.fit(df)
        X = self.feature_pipeline.transform(df)
        y = df[target_col]

        # 2. Supervised Model: XGBoost
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.xgb_model.fit(X, y)

        # 3. Unsupervised Model: Isolation Forest
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )
        self.iso_forest.fit(X)

        # Save artifacts
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'xgb': self.xgb_model,
                'iso': self.iso_forest,
                'pipeline': self.feature_pipeline
            }, f)

    def load(self):
        with open(self.model_path, 'rb') as f:
            artifacts = pickle.load(f)
            self.xgb_model = artifacts['xgb']
            self.iso_forest = artifacts['iso']
            self.feature_pipeline = artifacts['pipeline']

    def score(self, df):
        X = self.feature_pipeline.transform(df)
        
        # XGBoost probability (0 to 1)
        prob_xgb = self.xgb_model.predict_proba(X)[:, 1]
        
        # Isolation Forest anomaly score
        # decision_function returns lower values for more abnormal observations
        # We normalize it to a 0-1 risk score (higher = more risky)
        if_scores = self.iso_forest.decision_function(X)
        prob_if = 1.0 / (1.0 + np.exp(if_scores * 10)) # Sigmoid to normalize

        # Hybrid Score: Weighted average
        # We trust XGBoost more for known patterns, IsoForest for anomalies
        final_score = (prob_xgb * 0.7) + (prob_if * 0.3)
        return final_score

    def explain(self, df_row):
        # Explain a single transaction using SHAP
        X = self.feature_pipeline.transform(df_row)
        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer.shap_values(X)
        
        feature_names = self.feature_pipeline.get_feature_names()
        # Get values for the positive class (fraud)
        vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        
        # Pair feature names with their SHAP values and sort by absolute impact
        impacts = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)
        
        # Return top 3 drivers
        return impacts[:3]
