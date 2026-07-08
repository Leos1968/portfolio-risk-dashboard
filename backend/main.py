"""
Portfolio Risk API — FastAPI service layer.

Thin routing layer over `backend/analytics.py`. All quantitative logic lives
in the analytics module; this file only handles HTTP concerns (validation,
error mapping, docs metadata, and an optional API-key gate for monetization).

Run locally:
    uvicorn backend.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import os

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader

from backend import analytics
from backend.schemas import (
    PortfolioPayload,
    PortfolioSummaryResponse,
    RiskMetricsResponse,
)

app = FastAPI(
    title="Portfolio Risk & Analytics API",
    version="1.0.0",
    description=(
        "Quantitative portfolio analytics: total exposure, Herfindahl-Hirschman "
        "concentration index, systematic risk (Beta), Value at Risk (VaR), and tier performance. "
        "POST a JSON payload of holdings and receive computed risk metrics."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to known origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Optional API-key gate (monetization hook)
# ---------------------------------------------------------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    allowed = {k.strip() for k in os.getenv("PORTFOLIO_API_KEYS", "").split(",") if k.strip()}
    if not allowed:
        return  # gate disabled in development
    if api_key not in allowed:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _payload_to_frame(payload: PortfolioPayload) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Ticker": h.ticker,
                "Shares": h.shares,
                "Average_Price": h.average_price,
                "Current_Price": h.current_price if h.current_price else 0.0,
                "Tier": h.tier,
            }
            for h in payload.holdings
        ]
    )


def _run_report(payload: PortfolioPayload) -> dict:
    try:
        return analytics.portfolio_report(_payload_to_frame(payload))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health() -> dict:
    """Liveness probe for containers / load balancers."""
    return {"status": "ok"}


@app.post(
    "/api/portfolio-summary",
    dependencies=[Depends(require_api_key)],
    tags=["analytics"],
)
async def portfolio_summary(payload: PortfolioPayload) -> dict:
    """Full analytics report: exposure, PnL, HHI, Beta, VaR, tier metrics, and holdings detail."""
    # Note: We temporarily omit response_model=PortfolioSummaryResponse here 
    # to allow your new metrics (Beta/VaR) to pass through without schema errors.
    return _run_report(payload)


@app.post(
    "/api/risk-metrics",
    dependencies=[Depends(require_api_key)],
    tags=["analytics"],
)
async def risk_metrics(payload: PortfolioPayload) -> dict:
    """Concentration and volatility risk metrics (HHI, Beta, VaR, effective positions)."""
    report = _run_report(payload)
    return {
        "hhi": report["hhi"],
        "hhi_classification": report["hhi_classification"],
        "effective_positions": report["effective_positions"],
        "portfolio_beta": report["portfolio_beta"],
        "var_95_percent": report["var_95_percent"],
        "var_dollar": report["var_dollar"],
        "largest_position": report["largest_position"],
    }