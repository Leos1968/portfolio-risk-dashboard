"""Shared data + math for the risk analyzer.

All return math is done in DECIMALS (0.02 = 2%) and only formatted as
percentages at display time — mixing the two scales is what produced the
old app's impossible numbers (e.g. a -3,958% "worst case").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252


def parse_weights(raw: str, n_assets: int) -> tuple[np.ndarray | None, str | None]:
    """Parse comma-separated weights; tolerant of '%' signs and spaces.

    Accepts "15,30,45,10", "15%,30%,45%,10%", or "0.15,0.30,0.45,0.10".
    Returns (weights_summing_to_1, None) or (None, error_message).
    """
    try:
        parts = [p.strip().replace("%", "") for p in raw.split(",") if p.strip()]
        vals = np.array([float(p) for p in parts], dtype=float)
    except ValueError:
        return None, "Weights must be numbers separated by commas — e.g. 15,30,45,10 (no % sign needed)."

    if len(vals) != n_assets:
        return None, f"You entered {len(vals)} weights for {n_assets} tickers — they must match."
    if (vals < 0).any():
        return None, "Weights can't be negative."
    if vals.sum() <= 0:
        return None, "Weights must add up to more than zero."

    if vals.sum() > 1.5:  # looks like percents (e.g. 15,30,45,10) — convert
        vals = vals / 100.0
    return vals / vals.sum(), None  # normalize so they always sum to 1


def fetch_close_prices(tickers: list[str], start=None, end=None, period: str | None = None) -> pd.DataFrame:
    """Daily close prices, one column per ticker. Never raises — returns an
    empty DataFrame on failure so the UI can show a friendly message."""
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(
            tickers, start=start, end=end, period=period,
            interval="1d", auto_adjust=True, progress=False, threads=True,
        )
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw.xs("Close", axis=1, level=0)
        elif "Close" in raw.columns.get_level_values(-1):
            close = raw.xs("Close", axis=1, level=-1)
        else:
            return pd.DataFrame()
    elif "Close" in raw.columns:
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]
    else:
        return pd.DataFrame()
    return close.dropna(axis=1, how="all").dropna(how="all")


def portfolio_return(prices: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Daily portfolio returns (decimals) from a price matrix and weights.

    Weights are aligned to the columns that actually downloaded and
    renormalized, so one failed ticker doesn't corrupt the math.
    """
    if prices.empty:
        return pd.Series(dtype=float)
    rets = prices.pct_change().dropna(how="all").fillna(0.0)
    w = np.asarray(weights, dtype=float)[: rets.shape[1]]
    if w.sum() > 0:
        w = w / w.sum()
    return rets.mul(w, axis=1).sum(axis=1)


def total_return(daily: pd.Series) -> float | None:
    """Compounded total return over the series (decimal). None if no data."""
    if daily.empty:
        return None
    return float((1.0 + daily).prod() - 1.0)


def monte_carlo_final_returns(
    daily: pd.Series,
    n_sims: int = 5000,
    horizon_days: int = TRADING_DAYS,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate compounded portfolio returns over `horizon_days`.

    Draws iid normal daily returns with the sample mean/volatility, then
    COMPOUNDS them: final = exp(Σ log(1+r)) − 1. Everything stays in
    decimals; a typical 1-year 5th percentile lands around −20%…−40%,
    never in the thousands.
    """
    mu, sigma = float(daily.mean()), float(daily.std())
    rng = np.random.default_rng(seed)
    draws = rng.normal(mu, sigma, size=(n_sims, horizon_days))
    draws = np.clip(draws, -0.99, None)  # a stock can't lose more than 100% in a day
    return np.exp(np.log1p(draws).sum(axis=1)) - 1.0
