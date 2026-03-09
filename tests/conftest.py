import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Age': np.random.randint(25, 65, n_samples),
        'Annual_Income': np.random.uniform(20000, 150000, n_samples),
        'Loan_Amount': np.random.uniform(5000, 50000, n_samples),
        'Credit_History_Years': np.random.randint(0, 30, n_samples),
        'Employment_Years': np.random.randint(0, 40, n_samples),
        'Default': np.random.choice([0, 1], n_samples, p=[0.56, 0.44])
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def scaled_data(sample_data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled = scaler.fit_transform(sample_data.drop('Default', axis=1))
    return scaled, sample_data['Default'].values