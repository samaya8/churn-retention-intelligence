"""Unit tests for the customer churn preprocessor."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessor import ChurnPreprocessor, NUMERICAL_COLS


@pytest.fixture(scope="module")
def preprocessor() -> ChurnPreprocessor:
    return ChurnPreprocessor()


def test_preprocessor_initialization(preprocessor: ChurnPreprocessor) -> None:
    """Verify preprocessor loads scaler and feature columns properly."""
    assert preprocessor.scaler is not None
    assert preprocessor.feature_columns is not None
    assert len(preprocessor.feature_columns) == 13


def test_clean_record_defaults(preprocessor: ChurnPreprocessor) -> None:
    """Verify missing values are imputed with sensible defaults."""
    empty_dict = {}
    cleaned = preprocessor.clean_record(empty_dict)

    assert cleaned["CreditScore"] == 650.0
    assert cleaned["Geography"] == "France"
    assert cleaned["Gender"] == "Female"
    assert cleaned["Age"] == 40.0
    assert cleaned["Tenure"] == 5.0
    assert cleaned["Balance"] == 50000.0
    assert cleaned["NumOfProducts"] == 1
    assert cleaned["HasCrCard"] == 1
    assert cleaned["IsActiveMember"] == 1


def test_clean_record_normalization(preprocessor: ChurnPreprocessor) -> None:
    """Verify strings and boolean representations are normalized."""
    raw = {
        "Geography": "  germany ",
        "Gender": "m",
        "HasCrCard": "yes",
        "IsActiveMember": "false",
        "NumOfProducts": "2",
        "Age": "120",   # Out of bound clamped to 100
        "Tenure": "-5", # Out of bound clamped to 0
    }
    cleaned = preprocessor.clean_record(raw)

    assert cleaned["Geography"] == "Germany"
    assert cleaned["Gender"] == "Male"
    assert cleaned["HasCrCard"] == 1
    assert cleaned["IsActiveMember"] == 0
    assert cleaned["NumOfProducts"] == 2
    assert cleaned["Age"] == 100.0
    assert cleaned["Tenure"] == 0.0


def test_transform_single_shape_and_columns(preprocessor: ChurnPreprocessor) -> None:
    """Verify transform_single outputs 1 row aligned exactly with feature_columns."""
    sample = {
        "CreditScore": 700,
        "Geography": "Spain",
        "Gender": "Male",
        "Age": 36,
        "Tenure": 4,
        "Balance": 45000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 62000.0,
    }
    encoded = preprocessor.transform_single(sample)

    assert encoded.shape == (1, 13)
    assert list(encoded.columns) == preprocessor.feature_columns
    assert encoded["Geography_Spain"].iloc[0] == 1
    assert encoded["Geography_France"].iloc[0] == 0
    assert encoded["Gender_Male"].iloc[0] == 1


def test_transform_batch(preprocessor: ChurnPreprocessor) -> None:
    """Verify batch DataFrame transformation aligns columns and removes identifiers."""
    df = pd.DataFrame([
        {
            "RowNumber": 1,
            "CustomerId": 15634602,
            "Surname": "Hargrave",
            "CreditScore": 619,
            "Geography": "France",
            "Gender": "Female",
            "Age": 42,
            "Tenure": 2,
            "Balance": 0.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 101348.88,
            "Exited": 1,
        },
        {
            "RowNumber": 2,
            "CustomerId": 15647311,
            "Surname": "Hill",
            "CreditScore": 608,
            "Geography": "Spain",
            "Gender": "Female",
            "Age": 41,
            "Tenure": 1,
            "Balance": 83807.86,
            "NumOfProducts": 1,
            "HasCrCard": 0,
            "IsActiveMember": 1,
            "EstimatedSalary": 112542.58,
            "Exited": 0,
        },
    ])

    encoded = preprocessor.transform_batch(df)
    assert encoded.shape == (2, 13)
    assert "RowNumber" not in encoded.columns
    assert "CustomerId" not in encoded.columns
    assert "Surname" not in encoded.columns
    assert "Exited" not in encoded.columns
    assert list(encoded.columns) == preprocessor.feature_columns
