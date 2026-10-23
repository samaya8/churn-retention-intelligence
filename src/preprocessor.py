"""Preprocessing and feature transformation pipeline for customer churn data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCALER_PATH = PROJECT_DIR / "scaler.pkl"
DEFAULT_COLUMNS_PATH = PROJECT_DIR / "feature_columns.pkl"

NUMERICAL_COLS: List[str] = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
CATEGORICAL_COLS: List[str] = ["Geography", "Gender"]
BINARY_COLS: List[str] = ["HasCrCard", "IsActiveMember"]

DEFAULT_CUSTOMER_VALUES: Dict[str, Any] = {
    "CreditScore": 650.0,
    "Geography": "France",
    "Gender": "Female",
    "Age": 40.0,
    "Tenure": 5.0,
    "Balance": 50000.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 60000.0,
}


class ChurnPreprocessor:
    """Standardizes, encodes, and scales raw customer records for XGBoost model inference."""

    def __init__(
        self,
        scaler_path: Optional[Union[str, Path]] = None,
        feature_columns_path: Optional[Union[str, Path]] = None,
    ):
        s_path = Path(scaler_path) if scaler_path else DEFAULT_SCALER_PATH
        f_path = Path(feature_columns_path) if feature_columns_path else DEFAULT_COLUMNS_PATH

        self.scaler = joblib.load(s_path) if s_path.exists() else None
        self.feature_columns = list(joblib.load(f_path)) if f_path.exists() else None

    def clean_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize types and fill missing fields with defaults."""
        cleaned = dict(DEFAULT_CUSTOMER_VALUES)
        cleaned.update(record)

        # Normalize CreditScore
        cleaned["CreditScore"] = float(cleaned.get("CreditScore") or 650.0)

        # Normalize Geography
        geo = str(cleaned.get("Geography", "France")).strip().title()
        cleaned["Geography"] = geo if geo in ["France", "Germany", "Spain"] else "France"

        # Normalize Gender
        gender = str(cleaned.get("Gender", "Female")).strip().capitalize()
        if gender in ["M", "Male"]:
            cleaned["Gender"] = "Male"
        elif gender in ["F", "Female"]:
            cleaned["Gender"] = "Female"
        else:
            cleaned["Gender"] = "Female"

        # Normalize Age and Tenure
        cleaned["Age"] = float(max(18.0, min(100.0, float(cleaned.get("Age") or 40.0))))
        cleaned["Tenure"] = float(max(0.0, min(10.0, float(cleaned.get("Tenure") or 5.0))))

        # Normalize Balance & Salary
        cleaned["Balance"] = float(max(0.0, float(cleaned.get("Balance") or 0.0)))
        cleaned["EstimatedSalary"] = float(max(0.0, float(cleaned.get("EstimatedSalary") or 50000.0)))

        # Normalize NumOfProducts
        cleaned["NumOfProducts"] = int(max(1, min(4, int(cleaned.get("NumOfProducts") or 1))))

        # Normalize Binary flags
        for col in BINARY_COLS:
            val = cleaned.get(col, 1)
            if isinstance(val, str):
                cleaned[col] = 1 if val.strip().lower() in ["yes", "true", "1"] else 0
            else:
                cleaned[col] = 1 if bool(val) else 0

        return cleaned

    def transform_single(self, customer: Dict[str, Any]) -> pd.DataFrame:
        """Transform a single customer dictionary into an aligned and scaled DataFrame."""
        cleaned = self.clean_record(customer)
        df = pd.DataFrame([cleaned])
        return self.transform_batch(df)

    def transform_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode and scale a batch DataFrame, enforcing alignment with trained feature columns."""
        df_clean = df.copy()

        # Remove administrative / identifier columns if present
        for id_col in ["RowNumber", "CustomerId", "Surname", "Exited"]:
            if id_col in df_clean.columns:
                df_clean = df_clean.drop(columns=[id_col])

        # Fill missing values
        for col, default_val in DEFAULT_CUSTOMER_VALUES.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(default_val)
            else:
                df_clean[col] = default_val

        # Normalize binary columns
        for col in BINARY_COLS:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(
                    lambda v: 1 if str(v).strip().lower() in ["yes", "true", "1", "1.0"] else 0
                )

        # One-hot encode Geography and Gender
        encoded = pd.get_dummies(df_clean, columns=CATEGORICAL_COLS, drop_first=False)

        # Enforce column alignment
        if self.feature_columns is not None:
            encoded = encoded.reindex(columns=self.feature_columns, fill_value=0)

        # Scale numerical columns
        if self.scaler is not None:
            encoded[NUMERICAL_COLS] = self.scaler.transform(encoded[NUMERICAL_COLS])

        return encoded
