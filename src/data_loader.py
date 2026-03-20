"""
data_loader.py
--------------
Loads and validates the Stroke Prediction dataset.
Handles the quirk where BMI is stored as 'N/A' strings in the raw CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "healthcare-dataset-stroke-data.csv"

# ── expected schema ────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "id", "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status", "stroke",
]
TARGET_COLUMN = "stroke"


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw CSV file from *path*.

    The dataset stores missing BMI values as the literal string ``'N/A'``
    rather than an empty cell, so we pass ``na_values=['N/A']`` to let
    pandas convert them to proper ``NaN`` floats.

    Parameters
    ----------
    path : Path
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with proper NaN handling.

    Raises
    ------
    FileNotFoundError
        If the CSV is not found at *path*.
    ValueError
        If expected columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            f"Place the CSV file at: {DATA_PATH}"
        )

    df = pd.read_csv(path, na_values=["N/A"])

    # ── schema validation ──────────────────────────────────────────────────────
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    logger.info("Loaded %d rows × %d columns from %s", *df.shape, path.name)
    logger.info(
        "Target distribution:\n%s",
        df[TARGET_COLUMN].value_counts().to_string(),
    )

    return df


def get_feature_info(df: pd.DataFrame) -> dict:
    """
    Return a summary dict with shape, missing value counts, and class balance.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or preprocessed dataframe.

    Returns
    -------
    dict
        Keys: ``shape``, ``nulls``, ``class_balance``.
    """
    return {
        "shape": df.shape,
        "nulls": df.isnull().sum().to_dict(),
        "class_balance": df[TARGET_COLUMN].value_counts(normalize=True).to_dict()
        if TARGET_COLUMN in df.columns
        else {},
    }
