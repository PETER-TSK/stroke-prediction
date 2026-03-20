"""
tests/test_data_loader.py
-------------------------
Unit tests for src/data_loader.py.

Run with:
    pytest tests/test_data_loader.py -v

Tests write temporary CSV files using pytest's tmp_path fixture —
no real dataset is required.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# ── make src/ importable ──────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import load_raw, get_feature_info, REQUIRED_COLUMNS, TARGET_COLUMN


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_valid_csv(path: Path, n: int = 10) -> None:
    """Write a minimal but valid stroke CSV to *path*."""
    rows = [
        {
            "id": i,
            "gender": "Male" if i % 2 == 0 else "Female",
            "age": 40 + i,
            "hypertension": i % 2,
            "heart_disease": 0,
            "ever_married": "Yes",
            "work_type": "Private",
            "Residence_type": "Urban",
            "avg_glucose_level": 100.0 + i,
            "bmi": 25.0 if i % 3 != 0 else "N/A",  # some N/A values
            "smoking_status": "never smoked",
            "stroke": 1 if i == 0 else 0,
        }
        for i in range(n)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


# ── load_raw() ────────────────────────────────────────────────────────────────

class TestLoadRaw:
    def test_loads_valid_csv(self, tmp_path):
        csv = tmp_path / "data.csv"
        _write_valid_csv(csv)
        df = load_raw(csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

    def test_na_values_converted(self, tmp_path):
        """'N/A' strings in BMI column must become NaN."""
        csv = tmp_path / "data.csv"
        _write_valid_csv(csv, n=10)
        df = load_raw(csv)
        # Rows where bmi was 'N/A' (i % 3 == 0) should be NaN
        assert df["bmi"].isna().any()
        assert df["bmi"].dtype != object  # must be numeric

    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_raw(tmp_path / "nonexistent.csv")

    def test_raises_if_columns_missing(self, tmp_path):
        csv = tmp_path / "bad.csv"
        # Write a CSV with only 3 columns
        pd.DataFrame({"id": [1], "age": [50]}).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="missing expected columns"):
            load_raw(csv)

    def test_all_required_columns_present(self, tmp_path):
        csv = tmp_path / "data.csv"
        _write_valid_csv(csv)
        df = load_raw(csv)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Expected column '{col}' not found"

    def test_target_column_is_integer(self, tmp_path):
        csv = tmp_path / "data.csv"
        _write_valid_csv(csv)
        df = load_raw(csv)
        assert pd.api.types.is_integer_dtype(df[TARGET_COLUMN])

    def test_accepts_path_string(self, tmp_path):
        csv = tmp_path / "data.csv"
        _write_valid_csv(csv)
        # Should also work when path is passed as a string
        df = load_raw(str(csv))
        assert len(df) == 10

    def test_does_not_alter_non_bmi_strings(self, tmp_path):
        csv = tmp_path / "data.csv"
        _write_valid_csv(csv)
        df = load_raw(csv)
        # Gender values should still be strings
        assert df["gender"].dtype == object


# ── get_feature_info() ────────────────────────────────────────────────────────

class TestGetFeatureInfo:
    def test_returns_dict_with_expected_keys(self, tmp_path):
        csv = tmp_path / "data.csv"
        _write_valid_csv(csv)
        df = load_raw(csv)
        info = get_feature_info(df)
        assert "shape" in info
        assert "nulls" in info
        assert "class_balance" in info

    def test_shape_matches_dataframe(self, tmp_path):
        csv = tmp_path / "data.csv"
        _write_valid_csv(csv, n=15)
        df = load_raw(csv)
        info = get_feature_info(df)
        assert info["shape"] == df.shape

    def test_nulls_counts_bmi_correctly(self, tmp_path):
        csv = tmp_path / "data.csv"
        _write_valid_csv(csv, n=10)
        df = load_raw(csv)
        info = get_feature_info(df)
        # Rows with i % 3 == 0 are: 0, 3, 6, 9 → 4 missing BMI
        assert info["nulls"]["bmi"] == 4

    def test_class_balance_sums_to_one(self, tmp_path):
        csv = tmp_path / "data.csv"
        _write_valid_csv(csv)
        df = load_raw(csv)
        info = get_feature_info(df)
        total = sum(info["class_balance"].values())
        assert abs(total - 1.0) < 1e-9

    def test_class_balance_empty_if_no_target(self, tmp_path):
        csv = tmp_path / "data.csv"
        _write_valid_csv(csv)
        df = load_raw(csv).drop(columns=["stroke"])
        info = get_feature_info(df)
        assert info["class_balance"] == {}

    def test_nulls_dict_has_all_columns(self, tmp_path):
        csv = tmp_path / "data.csv"
        _write_valid_csv(csv)
        df = load_raw(csv)
        info = get_feature_info(df)
        assert set(info["nulls"].keys()) == set(df.columns)
