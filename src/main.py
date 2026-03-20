"""
main.py
-------
End-to-end orchestrator for the Stroke Prediction pipeline.

Usage
-----
    python src/main.py [--data PATH] [--time-limit SECONDS] [--preset PRESET]

Typical run (defaults):
    python src/main.py
    # → loads data from project root CSV
    # → cleans, splits, trains (≤3 min), evaluates, saves all artefacts
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── ensure src/ is importable regardless of CWD ──────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import data_loader
import evaluate as eval_module
import preprocess
import train as train_module

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stroke Prediction – AutoGluon pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=data_loader.DATA_PATH,
        help="Path to the raw CSV dataset.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=train_module.TIME_LIMIT,
        dest="time_limit",
        help="AutoGluon training budget in seconds.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=train_module.PRESET,
        choices=["best_quality", "high_quality", "good_quality", "medium_quality"],
        help="AutoGluon quality preset.",
    )
    return parser.parse_args()


# ── pipeline ──────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    """Execute the full ML pipeline end-to-end."""
    wall_start = time.perf_counter()

    print("\n" + "═" * 60)
    print("  Stroke Prediction  |  AutoGluon TabularPredictor")
    print("═" * 60 + "\n")

    # 1. Load ──────────────────────────────────────────────────────────────────
    logger.info("Step 1/4 – Loading data from %s", args.data)
    raw_df = data_loader.load_raw(args.data)
    info   = data_loader.get_feature_info(raw_df)
    logger.info("Dataset shape: %s", info["shape"])
    logger.info(
        "Class balance – No Stroke: %.1f%%  Stroke: %.1f%%",
        info["class_balance"].get(0, 0) * 100,
        info["class_balance"].get(1, 0) * 100,
    )

    # 2. Preprocess ────────────────────────────────────────────────────────────
    logger.info("Step 2/4 – Preprocessing")
    train_df, test_df = preprocess.run_pipeline(raw_df)

    # 3. Train ─────────────────────────────────────────────────────────────────
    logger.info(
        "Step 3/4 – Training (preset=%s, time_limit=%ds)",
        args.preset, args.time_limit,
    )
    predictor = train_module.train(
        train_df,
        time_limit=args.time_limit,
        preset=args.preset,
    )

    # 4. Evaluate ──────────────────────────────────────────────────────────────
    logger.info("Step 4/4 – Evaluating on held-out test set")
    metrics = eval_module.evaluate(predictor, test_df)

    # ── summary ───────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - wall_start
    print("\n" + "═" * 60)
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  F1-Stroke: {metrics['f1_stroke']:.4f}")
    print(f"  Results  : {train_module.RESULTS_DIR}")
    print(f"  Models   : {train_module.MODELS_DIR}")
    print("═" * 60 + "\n")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(parse_args())
