import pytest
import numpy as np
from src.model import CreditRiskModel
from src.validation import ModelValidator
from sklearn.datasets import make_classification

class TestCreditRiskModel:
    
    @pytest.fixture
    def setup(self):
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_train, X_test = X[:70], X[70:]
        y_train, y_test = y[:70], y[70:]
        return X_train, X_test, y_train, y_test
    
    def test_model_trains(self, setup):
        X_train, X_test, y_train, y_test = setup
        model = CreditRiskModel()
        model.train(X_train, y_train)
        assert model.train_auc is not None
        assert 0 < model.train_auc <= 1
    
    def test_model_predicts(self, setup):
        X_train, X_test, y_train, y_test = setup
        model = CreditRiskModel()
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert all(0 <= p <= 1 for p in predictions)
    
    def test_validation_auc(self, setup):
        X_train, X_test, y_train, y_test = setup
        model = CreditRiskModel()
        model.train(X_train, y_train)
        validator = ModelValidator()
        predictions = model.predict(X_test)
        auc = validator.calculate_auc(y_test, predictions)
        assert 0 < auc <= 1