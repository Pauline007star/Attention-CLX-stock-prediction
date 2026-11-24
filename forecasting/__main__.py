"""Run the next-close benchmark and save predictions with date and run metadata."""

import argparse
import hashlib
import json
import shutil
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from .data import ARIMA_ORDER, build_dataset, read_market


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Original single-stock CSV")
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--validation-start", default="2020-01-02")
    parser.add_argument("--test-start", default="2021-06-22")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N test targets")
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--baseline-only", action="store_true")
    models.add_argument("--xgboost", action="store_true", help="Include the optional residual tree")
    args = parser.parse_args()
    if args.epochs < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--epochs and --limit must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.output.exists():
        parser.error("--output must point to a new directory")
    return args


def expanding_arima(close, target_dates):
    predictions = []
    nonconverged = 0
    values = close.to_numpy()
    for position in close.index.get_indexer(target_dates):
        if position < 30:
            raise ValueError("Each ARIMA target must have at least 30 preceding sessions")
        fitted = ARIMA(values[:position], order=ARIMA_ORDER).fit()
        nonconverged += not fitted.mle_retvals["converged"]
        predictions.append(float(fitted.forecast()[0]))
    return predictions, nonconverged


def evaluate(predictions):
    actual = predictions["actual_close"]
    scores = {}
    for model in predictions.columns.drop(["as_of_date", "actual_close"]):
        forecast = predictions[model]
        scores[model] = {
            "mae": float(mean_absolute_error(actual, forecast)),
            "rmse": float(np.sqrt(mean_squared_error(actual, forecast))),
        }
    return scores


def run_experiment(args):
    data_digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    symbol, market = read_market(args.data)
    dataset = build_dataset(market, args.validation_start, args.test_start)
    targets = dataset.targets
    predictions = targets.loc[targets["split"] == "test", ["as_of_date", "actual_close", "last_close", "arima_fixed"]]
    predictions = predictions.iloc[:args.limit].copy()
    predictions["arima_210"], nonconverged = expanding_arima(market["close"], predictions.index)

    checkpoints = {}
    if not args.baseline_only:
        import torch

        from .models import fit_forecast

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        for name, windows in dataset.windows.items():
            forecast, checkpoints[name] = fit_forecast(name, windows, targets, args.epochs, args.seed)
            predictions[name] = forecast[:len(predictions)]

    if args.xgboost:
        from .trees import fit_residual_tree

        forecast, checkpoints["arima_xgboost"] = fit_residual_tree(
            dataset.windows["lstm_factors"][:, -6:], targets, args.seed
        )
        predictions["arima_xgboost"] = forecast[:len(predictions)]

    metrics = evaluate(predictions)
    packages = ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn"]
    if not args.baseline_only:
        packages.append("torch")
    if args.xgboost:
        packages.append("xgboost")
    metadata = {
        "stock": symbol,
        "data_sha256": data_digest,
        "validation_start": args.validation_start,
        "test_start": args.test_start,
        "counts": {**targets["split"].value_counts().to_dict(), "test_scored": len(predictions)},
        "last_test_target": predictions.index[-1].date().isoformat(),
        "seed": args.seed,
        "epochs": args.epochs,
        "arima_nonconverged_test_fits": nonconverged,
        "residual_initial_fit_converged": dataset.residual_fit_converged,
        "pca_explained_variance_ratio": dataset.pca_variance_ratio.tolist(),
        "validation_checkpoint_selection": checkpoints,
        "versions": {name: version(name) for name in packages},
        "metrics": metrics,
    }
    return predictions, metadata


def save_run(path, predictions, metadata):
    path.mkdir(parents=True, exist_ok=False)
    try:
        predictions.to_csv(path / "predictions.csv", float_format="%.17g")
        (path / "metrics.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    except Exception:
        shutil.rmtree(path)
        raise


def main():
    args = parse_args()
    predictions, metadata = run_experiment(args)
    save_run(args.output, predictions, metadata)
    print(json.dumps(metadata["metrics"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
