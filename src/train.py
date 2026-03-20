"""
train.py
--------
AutoGluon TabularPredictor training pipeline.

Strategy
~~~~~~~~
* Metric  : ``roc_auc``  – best for imbalanced binary classification.
* Preset  : ``'best_quality'`` with ``time_limit=180 s`` keeps total wall
            time under 5 min while still enabling stacking / bagging.
* GPU     : ``ag_args_fit={'num_gpus': 1}`` routes GPU-capable models
            (NN_TORCH, FASTAI) to the GPU; tree models run on CPU as normal.
* Verbosity: 1 (concise console output, no per-epoch noise).
* Outputs : saves leaderboard.csv + feature_importance.csv to ``results/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MODELS_DIR    = PROJECT_ROOT / "models"
RESULTS_DIR   = PROJECT_ROOT / "results"

TARGET        = "stroke"
EVAL_METRIC   = "roc_auc"
TIME_LIMIT    = 300          # seconds
PRESET        = "best_quality"


def train(
    train_df: pd.DataFrame,
    models_dir: Path = MODELS_DIR,
    time_limit: int = TIME_LIMIT,
    preset: str = PRESET,
) -> TabularPredictor:
    """
    Fit an AutoGluon ``TabularPredictor`` on *train_df*.

    AutoGluon automatically:
    * Selects and tunes multiple base learners (GBM, XGB, RF, NN, …).
    * Handles categorical encoding internally.
    * Balances class weights for imbalanced targets.
    * Stacks / bags models to maximise ROC-AUC.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split including the target column.
    models_dir : Path
        Directory where AutoGluon saves model artefacts.
    time_limit : int
        Wall-clock budget in seconds.
    preset : str
        AutoGluon quality preset (``'best_quality'``, ``'good_quality'``, etc.).

    Returns
    -------
    TabularPredictor
        Fitted predictor ready for inference.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Training AutoGluon predictor | preset=%s | time_limit=%ds | metric=%s | GPU=1",
        preset, time_limit, EVAL_METRIC,
    )

    predictor = TabularPredictor(
        label=TARGET,
        eval_metric=EVAL_METRIC,
        path=str(models_dir),
        verbosity=1,
    ).fit(
        train_data=train_df,
        presets=preset,
        time_limit=time_limit,
        # Route GPU-capable learners (NN_TORCH, FASTAI) to the GPU.
        # Tree-based models (GBM, XGB, RF, …) ignore this and run on CPU.
        ag_args_fit={"num_gpus": 1},
    )

    _save_leaderboard(predictor)
    _save_feature_importance(predictor, train_df)

    return predictor


def _save_leaderboard(predictor: TabularPredictor) -> None:
    """Persist the model leaderboard to ``results/leaderboard.csv``."""
    lb = predictor.leaderboard(silent=True)
    out = RESULTS_DIR / "leaderboard.csv"
    lb.to_csv(out, index=False)
    logger.info("Leaderboard saved → %s", out)
    logger.info("\n%s", lb[["model", "score_val", "fit_time", "pred_time_val"]].to_string(index=False))


def _save_feature_importance(
    predictor: TabularPredictor,
    train_df: pd.DataFrame,
) -> None:
    """Compute and persist feature importance to ``results/feature_importance.csv``."""
    try:
        fi = predictor.feature_importance(train_df, silent=True)
        out = RESULTS_DIR / "feature_importance.csv"
        fi.to_csv(out)
        logger.info("Feature importance saved → %s", out)
        logger.info("\nTop features:\n%s", fi.head(10).to_string())
    except Exception as exc:           # pragma: no cover
        logger.warning("Could not compute feature importance: %s", exc)


def load_predictor(models_dir: Path = MODELS_DIR) -> TabularPredictor:
    """
    Load a previously saved AutoGluon predictor from disk.

    Parameters
    ----------
    models_dir : Path
        Directory passed to :func:`train` as *models_dir*.

    Returns
    -------
    TabularPredictor
    """
    logger.info("Loading predictor from %s", models_dir)
    return TabularPredictor.load(str(models_dir))
