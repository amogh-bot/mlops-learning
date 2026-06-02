import joblib
import numpy as np
import pytest

@pytest.fixture
def model():
    return joblib.load("models/model.pkl")

def test_model_loads(model):
    assert model is not None

def test_prediction_shape(model):
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    proba = model.predict_proba(sample)
    assert proba.shape == (1, 3)

def test_probabilities_sum_to_one(model):
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    proba = model.predict_proba(sample)
    assert abs(proba.sum() - 1.0) < 1e-6

def test_known_setosa(model):
    setosa = np.array([[5.0, 3.4, 1.5, 0.2]])
    prediction = model.predict(setosa)[0]
    assert prediction == 0

def test_known_virginica(model):
    virginica = np.array([[7.7, 3.0, 6.1, 2.3]])
    prediction = model.predict(virginica)[0]
    assert prediction == 2