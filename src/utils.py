import json

def save_metrics(metrics, filename='model_metrics.json'):
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrikler kaydedildi: {filename}")

def load_metrics(filename='model_metrics.json'):
    with open(filename, 'r') as f:
        return json.load(f)

def print_model_summary(model_metrics):
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"AUC-ROC: {model_metrics.get('auc', 'N/A'):.4f}")
    print(f"Gini: {model_metrics.get('gini', 'N/A'):.4f}")
    print(f"KS Statistic: {model_metrics.get('ks', 'N/A'):.4f}")
    print("="*60 + "\n")