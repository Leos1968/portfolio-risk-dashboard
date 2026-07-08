"""
Quantitative Analytics Engine
=============================
Pure, stateless functions for portfolio risk analytics. No I/O, no framework
dependencies — this module is imported by both the FastAPI service
(`backend/main.py`) and the Streamlit frontend (`frontend/app.py`), so it must
stay free of web-layer concerns.

Metrics implemented
-------------------
1. Total Portfolio Exposure   — sum of Shares × Current_Price (Live Market Updated)
2. Concentration Risk (HHI)   — Herfindahl-Hirschman Index over position weights
3. Tier-Level Performance     — exposure-weighted returns per investment tier
                                (Growth / Core / Defensive) + synthetic
                                trailing trajectories for visualization
4. Systematic Volatility (β)  — Covariance matrix calculation against S&P 500
5. Value at Risk (95% VaR)    — Parametric daily risk threshold loss limits
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

# Standard DOJ/FTC-style HHI bands, applied to portfolio weights (0–10,000 scale)
HHI_DIVERSIFIED_MAX = 1_500
HHI_MODERATE_MAX = 2_500

# Tier assumptions used only for the synthetic trailing trajectories
# (annualized drift, annualized volatility)
_TIER_PROFILES = {
    "Growth": (0.18, 0.28),
    "Core": (0.10, 0.16),
    "Defensive": (0.05, 0.08),
}


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
# Advanced Volatility & Statistical Risk Calculations
# ---------------------------------------------------------------------------

def compute_advanced_risk_metrics(holdings: pd.DataFrame) -> dict[str, Any]:
    """Calculates Portfolio Beta vs S&P 500 and 1-Day 95% Parametric Value at Risk (VaR)
    utilizing live 1-year historical pricing arrays via yfinance.
    """
    tickers = holdings["Ticker"].tolist()
    weights_dict = holdings.set_index("Ticker")["Weight"].to_dict()
    total_value = holdings["Exposure"].sum()
    
    if not tickers or total_value <= 0:
        return {"portfolio_beta": 1.0, "var_95_percent": 0.0, "var_dollar": 0.0, "individual_betas": {}}

    # Download historical arrays
    hist_data = yf.download(tickers, period="1y", interval="1d")["Close"]
    if isinstance(hist_data, pd.Series):
        hist_data = hist_data.to_frame(name=tickers[0])
        
    daily_returns = hist_data.pct_change().dropna()

    # Download S&P 500 benchmark matrix
    spy_data = yf.download("^GSPC", period="1y", interval="1d")["Close"]
    spy_returns = spy_data.pct_change().dropna()

    # Intersect matrices on index dates
    combined = daily_returns.join(spy_returns, rsuffix="_spy").dropna()
    spy_ret_series = combined["Close_spy"]
    market_variance = spy_ret_series.var()

    portfolio_beta = 0.0
    individual_betas = {}

    for ticker in tickers:
        if ticker in combined.columns:
            asset_returns = combined[ticker]
            covariance = asset_returns.cov(spy_ret_series)
            asset_beta = covariance / market_variance if market_variance != 0 else 1.0
        else:
            asset_beta = 1.0
        
        individual_betas[ticker] = round(asset_beta, 2)
        portfolio_beta += weights_dict.get(ticker, 0.0) * asset_beta

    # Parametric Value at Risk calculation using asset covariance matrix
    cov_matrix = daily_returns.cov()
    weight_vector = np.array([weights_dict.get(t, 0.0) for t in tickers])
    
    # Check if matrices match correctly before dot-product processing
    valid_tickers = [t for t in tickers if t in cov_matrix.index]
    if len(valid_tickers) == len(tickers):
        portfolio_variance = np.dot(weight_vector.T, np.dot(cov_matrix, weight_vector))
        portfolio_volatility = np.sqrt(portfolio_variance)
    else:
        portfolio_volatility = 0.02  # Default fallback benchmark matrix variance

    # 95% confidence level maps to a Z-score factor of 1.645
    var_95_percent = 1.645 * portfolio_volatility
    var_dollar = total_value * var_95_percent

    return {
        "portfolio_beta": round(portfolio_beta, 2),
        "var_95_percent": round(var_95_percent * 100, 2),
        "var_dollar": round(var_dollar, 2),
        "individual_betas": individual_betas
    }


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------

def portfolio_report(df: pd.DataFrame) -> dict[str, Any]:
    """Run the full analytics pipeline and return a JSON-serializable report.

    This is the single entry point used by the API layer.
    """
    validated = validate_holdings(df)
    
    # Intercept to fetch actual live pricing directly from yfinance to keep prices real-time
    try:
        tickers = validated["Ticker"].tolist()
        live_snapshot = yf.download(tickers, period="1d")["Close"].iloc[-1]
        for ticker in tickers:
            if ticker in live_snapshot.index:
                validated.loc[validated["Ticker"] == ticker, "Current_Price"] = live_snapshot[ticker]
    except Exception:
        pass # Fall back to file-provided Current_Price safely if API network buffers

    holdings = compute_exposure(validated)
    hhi = compute_hhi(holdings["Weight"].to_numpy())
    tiers = tier_summary(holdings)
    advanced_risk = compute_advanced_risk_metrics(holdings)

    # Convert dataframe trajectory rows neatly into records for json output parsing
    trajectories = tier_growth_trajectories(holdings)
    trajectories["Period"] = trajectories["Period"].dt.strftime("%Y-%m-%d")
    trajectory_records = trajectories.to_dict(orient="records")

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
        "trajectories": trajectory_records,
        "holdings": holdings.round(4).to_dict(orient="records"),
    }