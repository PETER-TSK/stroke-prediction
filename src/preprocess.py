"""
preprocess.py
-------------
Cleans, encodes, and splits the stroke dataset.

Design choices
~~~~~~~~~~~~~~
* ``id``       – dropped (arbitrary identifier, zero predictive signal).
* ``gender``   – the single 'Other' sample is recoded to the mode ('Female')
                 to avoid sparse one-hot columns.
* ``bmi``      – 201 NaN values imputed with the **median** (robust to skew).
* Class imbalance (≈4.9 % positive) is left for AutoGluon to handle via its
  internal sample-weighting — it does this better than SMOTE for tree ensembles.
* Categorical string columns are left as ``object`` dtype so AutoGluon can
  apply its own optimised encoding; a scikit-learn fallback path is provided
  via ``encode_for_sklearn()``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

TARGET = "stroke"
DROP_COLS = ["id"]
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ── public API ─────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps and return a new dataframe.

    Steps
    -----
    1. Drop ``id`` column.
    2. Recode ``gender='Other'`` → mode.
    3. Impute missing ``bmi`` with median.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from :func:`data_loader.load_raw`.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe (copy, original unchanged).
    """
    df = df.copy()

    # 1. Drop identifier column
    df.drop(columns=DROP_COLS, inplace=True, errors="ignore")

    # 2. Recode rare 'Other' gender value
    gender_mode = df["gender"].mode()[0]
    other_count = (df["gender"] == "Other").sum()
    if other_count:
        df["gender"] = df["gender"].replace("Other", gender_mode)
        logger.info("Recoded %d 'Other' gender rows to '%s'", other_count, gender_mode)

    # 3. Impute BMI with median
    bmi_median = df["bmi"].median()
    missing_bmi = df["bmi"].isna().sum()
    df["bmi"] = df["bmi"].fillna(bmi_median)
    logger.info("Imputed %d missing BMI values with median=%.2f", missing_bmi, bmi_median)

    assert df.isnull().sum().sum() == 0, "Unexpected nulls remain after cleaning"

    logger.info("Clean shape: %d rows × %d cols", *df.shape)
    return df


def split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split.

    Stratification preserves the ~4.9 % positive rate in both splits.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (must contain the ``stroke`` target column).
    test_size : float
        Fraction of data held out for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)`` – each containing *all* columns including target.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET],
        random_state=random_state,
    )

    logger.info(
        "Split → train=%d (pos=%.1f%%)  test=%d (pos=%.1f%%)",
        len(train_df), 100 * train_df[TARGET].mean(),
        len(test_df),  100 * test_df[TARGET].mean(),
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def encode_for_sklearn(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode all ``object`` columns for scikit-learn compatibility.

    Encoders are fit on **train only** and applied to both splits.

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df  : pd.DataFrame

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, dict]
        Encoded train, encoded test, and a mapping of column → fitted encoder.
    """
    train_enc = train_df.copy()
    test_enc  = test_df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in train_df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        train_enc[col] = le.fit_transform(train_df[col].astype(str))
        test_enc[col]  = le.transform(test_df[col].astype(str))
        encoders[col]  = le

    return train_enc, test_enc, encoders


def run_pipeline(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: clean → split.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)`` ready for AutoGluon.
    """
    clean_df = clean(df)
    return split(clean_df)
