"""
predict.py
----------
Batch inference script: apply the trained model to new patient data.

Usage
-----
    # CSV input → CSV output
    python src/predict.py --input new_patients.csv --output predictions.csv

    # CSV input → print to console
    python src/predict.py --input new_patients.csv

    # Adjust decision threshold (lower = more sensitive, catches more strokes)
    python src/predict.py --input new_patients.csv --threshold 0.3

Input format
------------
    CSV file with any subset of the original feature columns (same names).
    The ``stroke`` column must NOT be present.
    The ``id`` column is optional — it is dropped automatically if present.
    Missing BMI values (NaN or the string ``'N/A'``) are imputed automatically
    using the training-set median.

Output columns
--------------
    * ``stroke_probability`` — model probability of stroke (0.0 – 1.0)
    * ``stroke_predicted``   — binary prediction at the chosen threshold
    * ``risk_level``         — human-readable label: Low / Medium / High

Example
-------
    python src/predict.py --input data/hospital_batch.csv \\
                          --output results/hospital_predictions.csv \\
                          --threshold 0.3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ── project paths ──────────────────────────────────────────────────────────────
SRC_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autogluon.tabular import TabularPredictor
from data_loader import load_raw, DATA_PATH

MODELS_DIR   = PROJECT_ROOT / "models"
DEFAULT_THRESHOLD = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predict")


# ── preprocessing ──────────────────────────────────────────────────────────────

def _get_training_bmi_median() -> float:
    """
    Return the BMI median computed on the full labelled dataset.

    Using the dataset median (rather than computing from inference data)
    avoids data leakage and matches what the model was trained on.
    """
    try:
        raw = load_raw(DATA_PATH)
        return float(raw["bmi"].median())
    except Exception:
        logger.warning(
            "Could not load training data to compute BMI median; "
            "falling back to population reference value 28.1"
        )
        return 28.1


def preprocess_for_inference(df: pd.DataFrame, bmi_median: float) -> pd.DataFrame:
    """
    Apply the same cleaning steps used during training.

    Steps
    -----
    1. Drop ``id`` if present (no predictive signal).
    2. Recode ``gender='Other'`` → ``'Female'`` (training-set mode).
    3. Convert BMI ``'N/A'`` strings to NaN, then impute with *bmi_median*.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data from the caller.
    bmi_median : float
        BMI median from the training set (used for imputation).

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for ``predictor.predict_proba()``.
    """
    df = df.copy()

    # 1. Drop id
    df.drop(columns=["id"], errors="ignore", inplace=True)

    # 2. Drop target if accidentally included
    df.drop(columns=["stroke"], errors="ignore", inplace=True)

    # 3. Recode rare gender value
    if "gender" in df.columns:
        df["gender"] = df["gender"].replace("Other", "Female")

    # 4. Handle BMI
    if "bmi" in df.columns:
        df["bmi"] = pd.to_numeric(df["bmi"].replace("N/A", float("nan")), errors="coerce")
        missing = df["bmi"].isna().sum()
        if missing:
            df["bmi"] = df["bmi"].fillna(bmi_median)
            logger.info("Imputed %d missing BMI value(s) with training median=%.2f", missing, bmi_median)

    return df


def _risk_label(prob: float, threshold: float) -> str:
    """Map a probability to a human-readable risk tier."""
    if prob < threshold * 0.5:
        return "Low"
    elif prob < threshold:
        return "Medium"
    else:
        return "High"


# ── prediction ─────────────────────────────────────────────────────────────────

def predict(
    input_path: Path,
    output_path: Path | None,
    threshold: float,
    models_dir: Path = MODELS_DIR,
) -> pd.DataFrame:
    """
    Load model, preprocess input, run inference, return results DataFrame.

    Parameters
    ----------
    input_path  : Path  CSV file containing new patient records.
    output_path : Path | None  If provided, results are saved here as CSV.
    threshold   : float  Decision boundary for binary prediction.
    models_dir  : Path  Directory containing saved AutoGluon artefacts.

    Returns
    -------
    pd.DataFrame
        Original columns plus ``stroke_probability``, ``stroke_predicted``,
        ``risk_level``.
    """
    # ── load model ─────────────────────────────────────────────────────────────
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {models_dir}\n"
            "Run 'python src/main.py' to train the model first."
        )
    logger.info("Loading predictor from %s", models_dir)
    predictor = TabularPredictor.load(str(models_dir))

    # ── load & validate input ──────────────────────────────────────────────────
    logger.info("Reading input from %s", input_path)
    raw = pd.read_csv(input_path, na_values=["N/A"])
    logger.info("Input shape: %d rows × %d columns", *raw.shape)

    if "stroke" in raw.columns:
        logger.warning(
            "Column 'stroke' found in input — it will be dropped. "
            "To evaluate against ground truth, use evaluate.py instead."
        )

    # ── preprocess ─────────────────────────────────────────────────────────────
    bmi_median  = _get_training_bmi_median()
    clean       = preprocess_for_inference(raw, bmi_median)

    # ── inference ──────────────────────────────────────────────────────────────
    logger.info("Running inference with threshold=%.2f …", threshold)
    proba = predictor.predict_proba(clean)
    y_score = proba[1].values if isinstance(proba, pd.DataFrame) else proba

    # ── build results ──────────────────────────────────────────────────────────
    results = raw.copy()
    results["stroke_probability"] = y_score.round(4)
    results["stroke_predicted"]   = (y_score >= threshold).astype(int)
    results["risk_level"]         = [_risk_label(p, threshold) for p in y_score]

    n_high = (results["stroke_predicted"] == 1).sum()
    logger.info(
        "Done. %d / %d patients flagged as high-risk (threshold=%.2f).",
        n_high, len(results), threshold,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        logger.info("Predictions saved → %s", output_path)

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stroke Prediction — batch inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to CSV file with patient records.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path to save predictions CSV. If omitted, prints to console.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Decision threshold for binary stroke prediction. "
            "Lower values increase sensitivity (catch more strokes but more false alarms). "
            "Typical clinical range: 0.2–0.4 for high-recall screening."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory containing saved AutoGluon artefacts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = predict(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        models_dir=args.models_dir,
    )
    if args.output is None:
        print(results.to_string(index=False))
