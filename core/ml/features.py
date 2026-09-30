import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class FeaturePipeline:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_cols = ['merchant_category', 'location', 'device_fingerprint']
        self.numerical_cols = ['amount', 'velocity_1h', 'velocity_24h', 'avg_amount_30d', 'distance_from_home_km']

    def fit(self, df):
        # Fit label encoders
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Fit scaler
        self.scaler.fit(df[self.numerical_cols])
        return self

    def transform(self, df):
        df_processed = df.copy()
        
        # Transform categories
        for col, le in self.label_encoders.items():
            # Handle unknown categories by mapping them to a default/new class
            df_processed[col] = df_processed[col].astype(str).map(
                lambda s: le.transform([s])[0] if s in le.classes_ else -1
            )
            
        # Transform numericals
        df_processed[self.numerical_cols] = self.scaler.transform(df_processed[self.numerical_cols])
        
        return df_processed[self.numerical_cols + self.categorical_cols]

    def get_feature_names(self):
        return self.numerical_cols + self.categorical_cols
