from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import numpy as np

class ModelValidator:
    def __init__(self):
        self.metrics = {}
    
    def calculate_auc(self, y_true, y_pred_proba):
        auc = roc_auc_score(y_true, y_pred_proba)
        self.metrics['auc'] = auc
        return auc
    
    def calculate_gini(self, y_true, y_pred_proba):
        auc = roc_auc_score(y_true, y_pred_proba)
        gini = 2 * auc - 1
        self.metrics['gini'] = gini
        return gini
    
    def calculate_ks_statistic(self, y_true, y_pred_proba):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        ks = np.max(tpr - fpr)
        self.metrics['ks'] = ks
        return ks
    
    def get_confusion_matrix(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0
        }