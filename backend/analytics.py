"""
Quantitative Analytics Engine
=============================
Pure, stateless functions for portfolio risk analytics. No I/O side effects
beyond market-data fetches, no framework dependencies — this module is
imported by both the FastAPI service (`backend/main.py`) and the Streamlit
frontend (`frontend/app.py`), so it must stay free of web-layer concerns.

Metrics implemented
-------------------
1. Total Portfolio Exposure   — sum of Shares × Current_Price (live-updated)
2. Concentration Risk (HHI)   — Herfindahl-Hirschman Index over position weights
3. Tier-Level Performance     — exposure-weighted returns per investment tier
                                (Growth / Core / Defensive) + synthetic
                                trailing trajectories for visualization
4. Systematic Volatility (β)  — covariance against the S&P 500 (^GSPC)
5. Value at Risk (95% VaR)    — parametric 1-day risk threshold

Market data
-----------
All Yahoo Finance access goes through `fetch_close_prices`, which never
raises: on rate limits, network failures, or unexpected response shapes it
returns an empty DataFrame and the analytics degrade gracefully (beta
defaults to 1.0, prices fall back to the uploaded CSV values). The report
carries a `market_data` field ("live" or "unavailable") so the UI can tell
the user which mode it is in.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ["Ticker", "Shares", "Average_Price", "Current_Price", "Tier"]
VALID_TIERS = ("Growth", "Core", "Defensive")

BENCHMARK = "^GSPC"  # S&P 500 index

# Standard DOJ/FTC-style HHI bands, applied to portfolio weights (0–10,000 scale)
HHI_DIVERSIFIED_MAX = 1_500
HHI_MODERATE_MAX = 2_500

# Z-score for a one-sided 95% confidence interval
_Z_95 = 1.645

# Conservative daily volatility assumed when market data is unavailable
_FALLBACK_DAILY_VOL = 0.02

# Tier assumptions used only for the synthetic trailing trajectories
# (annualized drift, annualized volatility)
_TIER_PROFILES = {
    "Growth": (0.18, 0.28),
    "Core": (0.10, 0.16),
    "Defensive": (0.05, 0.08),
}


# ---------------------------------------------------------------------------
# Market data (single hardened entry point for all yfinance access)
# ---------------------------------------------------------------------------

def fetch_close_prices(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Fetch daily close prices for `tickers`, one column per ticker.

    Never raises. Returns an empty DataFrame when Yahoo is unreachable,
    rate-limits the request, or returns an unexpected shape — callers must
    handle the empty case.
    """
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(
            tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    # Normalize the many column shapes yfinance can return:
    #  - MultiIndex (Price, Ticker) for multi-ticker downloads
    #  - MultiIndex (Ticker, Price) when group_by="ticker"
    #  - flat OHLCV columns for a single ticker
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

    close = close.dropna(axis=1, how="all").dropna(how="all")
    return close


def latest_prices(close: pd.DataFrame) -> pd.Series:
    """Most recent non-NaN close per ticker from a price matrix."""
    if close.empty:
        return pd.Series(dtype=float)
    return close.ffill().iloc[-1].dropna()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a holdings DataFrame.

    Ensures required columns exist, coerces numeric types, drops unusable
    rows, and normalizes tier labels. Returns a clean copy.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Holdings data is missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}"
        )

    clean = df[REQUIRED_COLUMNS].copy()
    clean["Ticker"] = clean["Ticker"].astype(str).str.strip().str.upper()
    clean["Tier"] = clean["Tier"].astype(str).str.strip().str.title()

    for col in ("Shares", "Average_Price", "Current_Price"):
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna(subset=["Shares", "Average_Price", "Current_Price"])
    clean = clean[(clean["Shares"] > 0) & (clean["Average_Price"] >= 0)]

    unknown = set(clean["Tier"]) - set(VALID_TIERS)
    if unknown:
        clean.loc[clean["Tier"].isin(unknown), "Tier"] = "Core"

    if clean.empty:
        raise ValueError("No valid holdings rows after validation.")

    return clean.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the holdings with per-position exposure and weights."""
    out = df.copy()
    out["Exposure"] = out["Shares"] * out["Current_Price"]
    out["Cost_Basis"] = out["Shares"] * out["Average_Price"]
    out["Unrealized_PnL"] = out["Exposure"] - out["Cost_Basis"]
    out["Return_Pct"] = np.where(
        out["Cost_Basis"] > 0,
        out["Unrealized_PnL"] / out["Cost_Basis"],
        0.0,
    )
    total = out["Exposure"].sum()
    out["Weight"] = out["Exposure"] / total if total > 0 else 0.0
    return out


def compute_hhi(weights: np.ndarray | pd.Series) -> float:
    """Herfindahl-Hirschman Index on the 0–10,000 scale."""
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    return float(np.sum((w * 100.0) ** 2))


def classify_hhi(hhi: float) -> str:
    """Map an HHI score to a qualitative concentration band."""
    if hhi < HHI_DIVERSIFIED_MAX:
        return "Diversified"
    if hhi < HHI_MODERATE_MAX:
        return "Moderately Concentrated"
    return "Highly Concentrated"


def effective_positions(hhi: float) -> float:
    """Effective number of independent positions implied by the HHI."""
    return 10_000.0 / hhi if hhi > 0 else 0.0


# ---------------------------------------------------------------------------
# Tier analytics
# ---------------------------------------------------------------------------

def tier_summary(df_exposed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exposure and weighted performance by investment tier."""
    def _agg(group: pd.DataFrame) -> pd.Series:
        exp = group["Exposure"].sum()
        w_ret = (
            float(np.average(group["Return_Pct"], weights=group["Exposure"]))
            if exp > 0
            else 0.0
        )
        return pd.Series(
            {
                "Exposure": exp,
                "Positions": len(group),
                "Weighted_Return": w_ret,
            }
        )

    summary = df_exposed.groupby("Tier").apply(_agg, include_groups=False)
    total = summary["Exposure"].sum()
    summary["Weight"] = summary["Exposure"] / total if total > 0 else 0.0
    order = [t for t in VALID_TIERS if t in summary.index]
    return summary.loc[order]


def tier_growth_trajectories(
    df_exposed: pd.DataFrame,
    periods: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic trailing growth trajectories per tier (indexed to 100)."""
    rng = np.random.default_rng(seed)
    tiers = tier_summary(df_exposed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=periods + 1, freq="ME")

    out = pd.DataFrame({"Period": dates})
    for tier, row in tiers.iterrows():
        drift, vol = _TIER_PROFILES.get(tier, _TIER_PROFILES["Core"])
        dt = 1.0 / 12.0
        shocks = rng.normal(drift * dt, vol * np.sqrt(dt), size=periods)
        path = np.concatenate([[0.0], np.cumsum(shocks)])
        target = np.log1p(row["Weighted_Return"])
        if abs(path[-1]) > 1e-9:
            path = path * (target / path[-1])
        out[tier] = 100.0 * np.exp(path)
    return out


# ---------------------------------------------------------------------------
# Advanced volatility & statistical risk
# ---------------------------------------------------------------------------

def compute_advanced_risk_metrics(
    holdings: pd.DataFrame,
    close_prices: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Portfolio Beta vs the S&P 500 and 1-day 95% parametric VaR.

    `close_prices` may be a pre-fetched matrix of daily closes (assets +
    benchmark) so callers can reuse a single download; when omitted, the
    data is fetched here. If market data is unavailable the function
    returns neutral defaults (beta 1.0, VaR from a conservative fallback
    volatility) instead of raising.
    """
    tickers = holdings["Ticker"].tolist()
    weights_dict = holdings.set_index("Ticker")["Weight"].to_dict()
    total_value = float(holdings["Exposure"].sum())

    defaults = {
        "portfolio_beta": 1.0,
        "var_95_percent": round(_Z_95 * _FALLBACK_DAILY_VOL * 100, 2),
        "var_dollar": round(total_value * _Z_95 * _FALLBACK_DAILY_VOL, 2),
        "individual_betas": {t: 1.0 for t in tickers},
        "market_data": "unavailable",
    }

    if not tickers or total_value <= 0:
        return {**defaults, "var_95_percent": 0.0, "var_dollar": 0.0}

    if close_prices is None:
        close_prices = fetch_close_prices(tickers + [BENCHMARK], period="1y")

    # Need at least a handful of observations for a meaningful covariance
    if close_prices.empty or len(close_prices) < 20:
        return defaults

    daily_returns = close_prices.pct_change().dropna(how="all")

    if BENCHMARK in daily_returns.columns:
        bench_returns = daily_returns[BENCHMARK].dropna()
        asset_returns = daily_returns.drop(columns=[BENCHMARK])
    else:
        bench_returns = pd.Series(dtype=float)
        asset_returns = daily_returns

    market_variance = bench_returns.var() if len(bench_returns) > 1 else np.nan
    have_benchmark = np.isfinite(market_variance) and market_variance > 0

    portfolio_beta = 0.0
    individual_betas = {}
    for ticker in tickers:
        if have_benchmark and ticker in asset_returns.columns:
            beta = asset_returns[ticker].cov(bench_returns) / market_variance
            if not np.isfinite(beta):
                beta = 1.0
        else:
            beta = 1.0
        individual_betas[ticker] = round(float(beta), 2)
        portfolio_beta += weights_dict.get(ticker, 0.0) * beta

    # Parametric VaR from the asset covariance matrix, aligned to the
    # tickers that actually downloaded
    cov_matrix = asset_returns.cov()
    valid_tickers = [t for t in tickers if t in cov_matrix.columns]

    if valid_tickers:
        weights_vec = np.array([weights_dict.get(t, 0.0) for t in valid_tickers])
        if weights_vec.sum() > 0:
            weights_vec = weights_vec / weights_vec.sum()
        sub_cov = cov_matrix.loc[valid_tickers, valid_tickers].to_numpy()
        port_var = float(weights_vec @ sub_cov @ weights_vec)
        volatility = np.sqrt(port_var) if port_var > 0 else _FALLBACK_DAILY_VOL
    else:
        volatility = _FALLBACK_DAILY_VOL

    return {
        "portfolio_beta": round(float(portfolio_beta), 2),
        "var_95_percent": round(float(_Z_95 * volatility * 100), 2),
        "var_dollar": round(float(total_value * _Z_95 * volatility), 2),
        "individual_betas": individual_betas,
        # "live" only if at least one asset (not just the benchmark) downloaded
        "market_data": "live" if valid_tickers else "unavailable",
    }


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------

def portfolio_report(df: pd.DataFrame) -> dict[str, Any]:
    """Run the full analytics pipeline and return a JSON-serializable report.

    This is the single entry point used by the API layer. One market-data
    download (assets + benchmark, 1y daily) feeds both the live price
    refresh and the beta/VaR calculations; if it fails, the report is
    built from the prices supplied in the uploaded holdings.
    """
    validated = validate_holdings(df)
    tickers = validated["Ticker"].tolist()

    close_prices = fetch_close_prices(tickers + [BENCHMARK], period="1y")

    live = latest_prices(close_prices)
    for ticker in tickers:
        price = live.get(ticker)
        if price is not None and np.isfinite(price) and price > 0:
            validated.loc[validated["Ticker"] == ticker, "Current_Price"] = float(price)

    holdings = compute_exposure(validated)
    hhi = compute_hhi(holdings["Weight"].to_numpy())
    tiers = tier_summary(holdings)
    advanced_risk = compute_advanced_risk_metrics(holdings, close_prices)

    trajectories = tier_growth_trajectories(holdings)
    trajectories["Period"] = trajectories["Period"].dt.strftime("%Y-%m-%d")

    return {
        "total_exposure": float(holdings["Exposure"].sum()),
        "total_cost_basis": float(holdings["Cost_Basis"].sum()),
        "total_unrealized_pnl": float(holdings["Unrealized_PnL"].sum()),
        "position_count": int(len(holdings)),
        "hhi": round(hhi, 2),
        "hhi_classification": classify_hhi(hhi),
        "effective_positions": round(effective_positions(hhi), 2),
        "portfolio_beta": advanced_risk["portfolio_beta"],
        "var_95_percent": advanced_risk["var_95_percent"],
        "var_dollar": advanced_risk["var_dollar"],
        "individual_betas": advanced_risk["individual_betas"],
        "market_data": advanced_risk["market_data"],
        "largest_position": {
            "ticker": str(holdings.loc[holdings["Weight"].idxmax(), "Ticker"]),
            "weight": float(holdings["Weight"].max()),
        },
        "tiers": {
            tier: {
                "exposure": float(row["Exposure"]),
                "weight": float(row["Weight"]),
                "positions": int(row["Positions"]),
                "weighted_return": float(row["Weighted_Return"]),
            }
            for tier, row in tiers.iterrows()
        },
        "trajectories": trajectories.to_dict(orient="records"),
        "holdings": holdings.round(4).to_dict(orient="records"),
    }
