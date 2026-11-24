"""Build feature windows whose target is the session after the window ends."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

MARKET_COLUMNS = ["open", "high", "low", "close", "vol", "amount"]
ARIMA_ORDER = (2, 1, 0)


@dataclass
class Dataset:
    windows: dict
    targets: pd.DataFrame
    residual_fit_converged: bool
    pca_variance_ratio: np.ndarray


def read_market(path):
    columns = ["ts_code", "trade_date", *MARKET_COLUMNS]
    frame = pd.read_csv(path, usecols=columns, dtype={"ts_code": str, "trade_date": str})
    if frame["ts_code"].isna().any() or frame["ts_code"].nunique() != 1:
        raise ValueError("Input must contain exactly one stock symbol")
    dates = pd.to_datetime(frame["trade_date"], format="%Y%m%d", errors="raise")
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Expected non-missing trading dates in strictly increasing order")
    market = frame[MARKET_COLUMNS].apply(pd.to_numeric).astype(float)
    if not np.isfinite(market).to_numpy().all():
        raise ValueError("Market fields must contain only finite values")
    if (market[["open", "high", "low", "close"]] <= 0).to_numpy().any():
        raise ValueError("Expected positive prices")
    if (market[["vol", "amount"]] < 0).to_numpy().any():
        raise ValueError("Expected nonnegative volume and trading amount")
    market.index = pd.DatetimeIndex(dates, name="trade_date")
    return frame["ts_code"].iloc[0], market


def causal_residual(close, warmup):
    """Estimate ARIMA parameters once, then use one-step filtered predictions."""
    values = close.to_numpy()
    initial_fit = ARIMA(values[:warmup], order=ARIMA_ORDER).fit()
    filtered = ARIMA(values, order=ARIMA_ORDER).filter(initial_fit.params)
    one_step_close = filtered.fittedvalues
    residual = values - one_step_close
    residual[:warmup] = np.nan
    innovations = pd.Series(residual, index=close.index, name="arima_residual")
    return innovations, one_step_close, bool(initial_fit.mle_retvals["converged"])


def build_factors(market, residual):
    close = market["close"]
    returns = close.pct_change()
    return pd.DataFrame({
        "return_1d": returns,
        "close_to_sma_5": close / close.rolling(5).mean() - 1,
        "volatility_10": returns.rolling(10).std(),
        "intraday_range": (market["high"] - market["low"]) / close,
        "log_volume_change": np.log1p(market["vol"]).diff(),
        "arima_residual": residual,
    })


def make_windows(values, target_positions, length):
    return np.stack([values[t - length:t] for t in target_positions]).astype(np.float32)


def build_dataset(market, validation_start, test_start, warmup=252):
    validation_start = pd.Timestamp(validation_start)
    test_start = pd.Timestamp(test_start)
    close = market["close"]
    dates = close.index
    if validation_start not in dates or test_start not in dates or validation_start >= test_start:
        raise ValueError("Choose two observed dates in validation-then-test order")

    if warmup < 30:
        raise ValueError("Need at least 30 observations for ARIMA warm-up")
    first_target = warmup + 20
    positions = np.arange(first_target, len(close))
    target_dates = dates[positions]
    is_train = target_dates < validation_start
    is_validation = (target_dates >= validation_start) & (target_dates < test_start)
    if is_train.sum() < 30 or not is_validation.any():
        raise ValueError("Need 30 or more training targets, plus validation observations")

    residual, fixed_arima_close, converged = causal_residual(close, warmup)
    factors = build_factors(market, residual).iloc[warmup:]
    factor_train_end = positions[is_train][-1] - warmup
    factor_train = factors.iloc[:factor_train_end]
    factor_scaler = StandardScaler().fit(factor_train)
    pca = PCA(n_components=2, svd_solver="full").fit(factor_scaler.transform(factor_train))
    transformed = pca.transform(factor_scaler.transform(factors))
    raw = market.to_numpy()[warmup:]
    relative_positions = positions - warmup
    views = {
        "lstm_market": (raw, 10),
        "lstm_factors": (np.column_stack([raw, factors.to_numpy()]), 10),
        "lstm_pca": (np.column_stack([raw, transformed]), 10),
        "attention_backbone": (np.column_stack([raw, residual.to_numpy()[warmup:]]), 20),
    }
    windows = {
        name: make_windows(values, relative_positions, length)
        for name, (values, length) in views.items()
    }
    targets = pd.DataFrame({
        "as_of_date": dates[positions - 1],
        "actual_close": close.to_numpy()[positions],
        "last_close": close.to_numpy()[positions - 1],
        "arima_fixed": fixed_arima_close[positions],
        "split": np.select([is_train, is_validation], ["train", "validation"], default="test"),
    }, index=target_dates)
    return Dataset(windows, targets, converged, pca.explained_variance_ratio_)
