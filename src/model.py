from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

class CreditRiskModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        self.train_auc = None
    
    def train(self, X_train, y_train):
        print("🤖 Model eğitiliyor...")
        self.model.fit(X_train, y_train)
        y_train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        self.train_auc = roc_auc_score(y_train, y_train_pred_proba)
        print(f"✅ Model eğitildi - Training AUC: {self.train_auc:.4f}")
        return self
    
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, feature_names):
        importances = {}
        for name, coef in zip(feature_names, self.model.coef_[0]):
            importances[name] = float(coef)
        return dict(sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True))