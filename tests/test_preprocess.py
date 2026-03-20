"""
tests/test_preprocess.py
------------------------
Unit tests for src/preprocess.py.

Run with:
    pytest tests/test_preprocess.py -v

These tests use synthetic DataFrames — no real dataset or model required.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── make src/ importable ──────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocess import clean, split, encode_for_sklearn


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a minimal synthetic stroke dataset for testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id":                rng.integers(1, 9999, n),
            "gender":            rng.choice(["Male", "Female"], n),
            "age":               rng.uniform(20, 80, n),
            "hypertension":      rng.integers(0, 2, n),
            "heart_disease":     rng.integers(0, 2, n),
            "ever_married":      rng.choice(["Yes", "No"], n),
            "work_type":         rng.choice(["Private", "Self-employed", "Govt_job"], n),
            "Residence_type":    rng.choice(["Urban", "Rural"], n),
            "avg_glucose_level": rng.uniform(70, 200, n),
            "bmi":               rng.uniform(18, 40, n),
            "smoking_status":    rng.choice(["never smoked", "formerly smoked", "smokes"], n),
            "stroke":            rng.choice([0, 1], n, p=[0.87, 0.13]),
        }
    )


# ── clean() ───────────────────────────────────────────────────────────────────

class TestClean:
    def test_drops_id_column(self):
        df = _make_df()
        assert "id" in df.columns
        result = clean(df)
        assert "id" not in result.columns

    def test_does_not_mutate_original(self):
        df = _make_df()
        original_cols = list(df.columns)
        clean(df)
        assert list(df.columns) == original_cols

    def test_no_nulls_after_clean(self):
        df = _make_df()
        # Inject some missing BMI values
        df.loc[:10, "bmi"] = np.nan
        result = clean(df)
        assert result.isnull().sum().sum() == 0

    def test_bmi_imputed_with_median(self):
        df = _make_df(n=200)
        expected_median = df["bmi"].median()
        missing_idx = [0, 1, 2, 3, 4, 5]
        df.loc[missing_idx, "bmi"] = np.nan
        result = clean(df)
        # No NaNs should remain
        assert not result["bmi"].isna().any()
        # Each formerly-NaN row should now hold exactly the pre-imputation median
        for i in missing_idx:
            assert abs(result.loc[i, "bmi"] - expected_median) < 0.01

    def test_gender_other_recoded(self):
        df = _make_df(n=100)
        # Inject one 'Other' gender row
        df.loc[0, "gender"] = "Other"
        result = clean(df)
        assert "Other" not in result["gender"].values

    def test_gender_other_recoded_to_mode(self):
        df = _make_df(n=100)
        mode = df["gender"].mode()[0]
        df.loc[0, "gender"] = "Other"
        result = clean(df)
        # The 'Other' row should be the mode value
        assert result.loc[0, "gender"] == mode

    def test_no_gender_other_leaves_column_unchanged(self):
        df = _make_df()
        assert "Other" not in df["gender"].values
        result = clean(df)
        # Should still run without error
        assert set(result["gender"].unique()).issubset({"Male", "Female"})

    def test_output_shape(self):
        df = _make_df(n=150)
        result = clean(df)
        # One column dropped (id), everything else preserved
        assert result.shape == (150, df.shape[1] - 1)

    def test_target_column_preserved(self):
        df = _make_df()
        result = clean(df)
        assert "stroke" in result.columns


# ── split() ───────────────────────────────────────────────────────────────────

class TestSplit:
    def test_returns_two_dataframes(self):
        df = clean(_make_df())
        train, test = split(df)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_split_sizes(self):
        df = clean(_make_df(n=500))
        train, test = split(df, test_size=0.2)
        assert len(train) + len(test) == len(df)
        # Test set is roughly 20% (±1 row due to stratification rounding)
        assert abs(len(test) - 100) <= 2

    def test_no_row_overlap(self):
        df = clean(_make_df(n=200))
        # Tag each row with a unique id before splitting
        df = df.copy()
        df["_uid"] = range(len(df))
        train, test = split(df)
        train_ids = set(train["_uid"])
        test_ids  = set(test["_uid"])
        assert train_ids.isdisjoint(test_ids)

    def test_stratified_positive_rate(self):
        df = clean(_make_df(n=1000, seed=0))
        original_rate = df["stroke"].mean()
        train, test = split(df, test_size=0.2)
        # Both splits should be within 3% of the original positive rate
        assert abs(train["stroke"].mean() - original_rate) < 0.03
        assert abs(test["stroke"].mean()  - original_rate) < 0.03

    def test_reset_index(self):
        df = clean(_make_df(n=100))
        train, test = split(df)
        assert list(train.index) == list(range(len(train)))
        assert list(test.index)  == list(range(len(test)))

    def test_reproducible_with_same_seed(self):
        df = clean(_make_df(n=200))
        train_a, test_a = split(df, random_state=42)
        train_b, test_b = split(df, random_state=42)
        pd.testing.assert_frame_equal(train_a, train_b)
        pd.testing.assert_frame_equal(test_a,  test_b)

    def test_different_seeds_give_different_splits(self):
        df = clean(_make_df(n=200))
        _, test_a = split(df, random_state=0)
        _, test_b = split(df, random_state=1)
        # Very unlikely to be identical
        assert not test_a.equals(test_b)


# ── encode_for_sklearn() ──────────────────────────────────────────────────────

class TestEncodeForSklearn:
    def test_no_object_columns_remain(self):
        df = clean(_make_df(n=200))
        train, test = split(df)
        train_enc, test_enc, _ = encode_for_sklearn(train, test)
        assert train_enc.select_dtypes(include="object").empty
        assert test_enc.select_dtypes(include="object").empty

    def test_returns_three_items(self):
        df = clean(_make_df())
        train, test = split(df)
        result = encode_for_sklearn(train, test)
        assert len(result) == 3

    def test_encoder_keys_match_object_columns(self):
        df = clean(_make_df(n=200))
        train, test = split(df)
        obj_cols = set(train.select_dtypes(include="object").columns)
        _, _, encoders = encode_for_sklearn(train, test)
        assert set(encoders.keys()) == obj_cols

    def test_shape_preserved(self):
        df = clean(_make_df(n=200))
        train, test = split(df)
        train_enc, test_enc, _ = encode_for_sklearn(train, test)
        assert train_enc.shape == train.shape
        assert test_enc.shape  == test.shape

    def test_does_not_mutate_inputs(self):
        df = clean(_make_df(n=200))
        train, test = split(df)
        train_dtypes_before = train.dtypes.copy()
        encode_for_sklearn(train, test)
        pd.testing.assert_series_equal(train.dtypes, train_dtypes_before)

    def test_numeric_columns_unchanged(self):
        df = clean(_make_df(n=200))
        train, test = split(df)
        train_enc, _, _ = encode_for_sklearn(train, test)
        for col in ["age", "avg_glucose_level", "bmi"]:
            pd.testing.assert_series_equal(
                train[col].reset_index(drop=True),
                train_enc[col].reset_index(drop=True),
            )
