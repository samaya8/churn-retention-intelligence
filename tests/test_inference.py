from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent

NUMERICAL_COLS = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]


@pytest.fixture(scope="module")
def artifacts():
    model = joblib.load(PROJECT_DIR / "xgb_churn_model.pkl")
    scaler = joblib.load(PROJECT_DIR / "scaler.pkl")
    feature_columns = joblib.load(PROJECT_DIR / "feature_columns.pkl")
    return model, scaler, feature_columns


def test_artifacts_exist():
    assert (PROJECT_DIR / "xgb_churn_model.pkl").exists()
    assert (PROJECT_DIR / "scaler.pkl").exists()
    assert (PROJECT_DIR / "feature_columns.pkl").exists()
    assert (PROJECT_DIR / "Churn_Modelling.csv").exists()


def test_feature_pipeline_inference(artifacts):
    model, scaler, feature_columns = artifacts
    
    sample_input = pd.DataFrame([{
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0,
    }])
    
    encoded = pd.get_dummies(sample_input, columns=["Geography", "Gender"], drop_first=False)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    encoded[NUMERICAL_COLS] = scaler.transform(encoded[NUMERICAL_COLS])
    
    pred = model.predict(encoded)
    prob = model.predict_proba(encoded)
    
    assert pred.shape == (1,)
    assert pred[0] in [0, 1]
    assert prob.shape == (1, 2)
    assert 0.0 <= prob[0][1] <= 1.0
