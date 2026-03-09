import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
    
    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Veri yüklendi: {self.df.shape[0]} satır, {self.df.shape[1]} sütun")
        return self.df
    
    def check_quality(self):
        quality = {
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'shape': self.df.shape
        }
        print(f"📊 Veri Kalitesi:")
        print(f"  - Kayıp değerler: {quality['missing_values']}")
        print(f"  - Tekrar eden satırlar: {quality['duplicates']}")
        return quality
    
    def prepare_for_modeling(self, target_column='Default'):
        X = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"✅ Train/Test split: Training {len(self.X_train)}, Testing {len(self.X_test)}")
        
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        return X_train_scaled, X_test_scaled, self.y_train, self.y_test