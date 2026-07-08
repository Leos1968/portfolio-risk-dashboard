"""
Quantitative Analytics Engine
=============================
Pure, stateless functions for portfolio risk analytics. No I/O, no framework
dependencies — this module is imported by both the FastAPI service
(`backend/main.py`) and the Streamlit frontend (`frontend/app.py`), so it must
stay free of web-layer concerns.

Metrics implemented
-------------------
1. Total Portfolio Exposure   — sum of Shares × Current_Price
2. Concentration Risk (HHI)   — Herfindahl-Hirschman Index over position weights
3. Tier-Level Performance     — exposure-weighted returns per investment tier
                                (Growth / Core / Defensive) + synthetic
                                trailing trajectories for visualization
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

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

    Raises
    ------
    ValueError
        If required columns are missing or no valid rows remain.
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
    clean = clean[(clean["Shares"] > 0) & (clean["Current_Price"] > 0)]

    unknown = set(clean["Tier"]) - set(VALID_TIERS)
    if unknown:
        # Unknown tiers are folded into Core rather than rejected — keeps
        # user uploads forgiving while preserving the three-tier model.
        clean.loc[clean["Tier"].isin(unknown), "Tier"] = "Core"

    if clean.empty:
        raise ValueError("No valid holdings rows after validation.")

    return clean.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the holdings with per-position exposure and weights.

    Adds columns:
        Exposure       — Shares × Current_Price
        Cost_Basis     — Shares × Average_Price
        Unrealized_PnL — Exposure − Cost_Basis
        Return_Pct     — position return vs. cost basis
        Weight         — Exposure / total exposure (0–1)
    """
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
    """Herfindahl-Hirschman Index on the 0–10,000 scale.

    HHI = Σ (wᵢ × 100)²  where wᵢ are portfolio weights summing to 1.
    10,000 = single-asset portfolio; 10,000 / N = perfectly equal N-asset book.
    """
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()  # re-normalize defensively
    return float(np.sum((w * 100.0) ** 2))


def classify_hhi(hhi: float) -> str:
    """Map an HHI score to a qualitative concentration band."""
    if hhi < HHI_DIVERSIFIED_MAX:
        return "Diversified"
    if hhi < HHI_MODERATE_MAX:
        return "Moderately Concentrated"
    return "Highly Concentrated"


def effective_positions(hhi: float) -> float:
    """Effective number of independent positions implied by the HHI (10,000 / HHI)."""
    return 10_000.0 / hhi if hhi > 0 else 0.0


# ---------------------------------------------------------------------------
# Tier analytics
# ---------------------------------------------------------------------------

def tier_summary(df_exposed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exposure and weighted performance by investment tier.

    Parameters
    ----------
    df_exposed : DataFrame produced by :func:`compute_exposure`.

    Returns
    -------
    DataFrame indexed by Tier with columns:
        Exposure, Weight, Positions, Weighted_Return
    """
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
    # Present tiers in canonical order, skipping any that are absent
    order = [t for t in VALID_TIERS if t in summary.index]
    return summary.loc[order]


def tier_growth_trajectories(
    df_exposed: pd.DataFrame,
    periods: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic trailing growth trajectories per tier (indexed to 100).

    Each tier's path is a deterministic geometric random walk whose drift and
    volatility come from the tier's risk profile, *anchored* so the final
    point equals the tier's actual weighted return computed from the holdings.
    This gives a realistic-looking trailing chart that is still consistent
    with the real end-state performance of the uploaded portfolio.

    Returns
    -------
    DataFrame with a monthly ``Period`` column plus one indexed-value column
    per tier present in the portfolio.
    """
    rng = np.random.default_rng(seed)
    tiers = tier_summary(df_exposed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=periods + 1, freq="ME")

    out = pd.DataFrame({"Period": dates})
    for tier, row in tiers.iterrows():
        drift, vol = _TIER_PROFILES.get(tier, _TIER_PROFILES["Core"])
        dt = 1.0 / 12.0
        shocks = rng.normal(drift * dt, vol * np.sqrt(dt), size=periods)
        path = np.concatenate([[0.0], np.cumsum(shocks)])
        # Anchor: rescale the path so its endpoint matches the realized return
        target = np.log1p(row["Weighted_Return"])
        if abs(path[-1]) > 1e-9:
            path = path * (target / path[-1])
        out[tier] = 100.0 * np.exp(path)
    return out


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------

def portfolio_report(df: pd.DataFrame) -> dict[str, Any]:
    """Run the full analytics pipeline and return a JSON-serializable report.

    This is the single entry point used by the API layer.
    """
    holdings = compute_exposure(validate_holdings(df))
    hhi = compute_hhi(holdings["Weight"].to_numpy())
    tiers = tier_summary(holdings)

    return {
        "total_exposure": float(holdings["Exposure"].sum()),
        "total_cost_basis": float(holdings["Cost_Basis"].sum()),
        "total_unrealized_pnl": float(holdings["Unrealized_PnL"].sum()),
        "position_count": int(len(holdings)),
        "hhi": round(hhi, 2),
        "hhi_classification": classify_hhi(hhi),
        "effective_positions": round(effective_positions(hhi), 2),
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
        "holdings": holdings.round(4).to_dict(orient="records"),
    }
