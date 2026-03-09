import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import pickle
import json
import os
from datetime import datetime

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("="*80)
print("CREDIT RISK MODEL - TRAINING")
print("="*80)

print("\n📊 Veri yükleniyor...")
df = pd.read_csv('data/sample_data.csv')
print(f"✅ {len(df)} satır veri yüklendi")
print(f"   Sütunlar: {list(df.columns)}")

print("\n🔧 Veri hazırlanıyor...")
X = df.drop('Default', axis=1)
y = df['Default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

print("\n🤖 Model eğitiliyor...")
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)
print("✅ Model eğitildi!")

print("\n📈 Tahmin yapılıyor...")
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ks = np.max(tpr - fpr)
gini = 2 * auc - 1

print(f"✅ Metrikler:")
print(f"   AUC: {auc:.4f}")
print(f"   Gini: {gini:.4f}")
print(f"   KS: {ks:.4f}")

print("\n💾 Model kaydediliyor...")
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ model.pkl ve scaler.pkl kaydedildi")

metrics = {
    'auc': float(auc),
    'gini': float(gini),
    'ks': float(ks),
    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('results/model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("✅ results/model_metrics.json kaydedildi")

print("\n" + "="*80)
print("✅ TAMAMLANDI! Model eğitildi ve kaydedildi.")
print("="*80)