"""
Pydantic request/response schemas for the Portfolio Risk API.

Keeping schemas in their own module keeps `main.py` focused on routing and
makes the contract easy to locate for API consumers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Holding(BaseModel):
    """A single portfolio position."""

    ticker: str = Field(..., min_length=1, max_length=12, description="Ticker symbol")
    shares: float = Field(..., gt=0, description="Number of shares held")
    average_price: float = Field(..., gt=0, description="Average acquisition price")
    current_price: float = Field(..., gt=0, description="Latest market price")
    tier: str = Field("Core", description="Investment tier: Growth | Core | Defensive")

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("tier")
    @classmethod
    def _tier(cls, v: str) -> str:
        t = v.strip().title()
        return t if t in ("Growth", "Core", "Defensive") else "Core"


class PortfolioPayload(BaseModel):
    """Request body: a list of holdings."""

    holdings: list[Holding] = Field(..., min_length=1, max_length=1000)


class TierMetrics(BaseModel):
    exposure: float
    weight: float
    positions: int
    weighted_return: float


class LargestPosition(BaseModel):
    ticker: str
    weight: float


class RiskMetricsResponse(BaseModel):
    """Response for /api/risk-metrics — concentration analytics only."""

    hhi: float
    hhi_classification: str
    effective_positions: float
    largest_position: LargestPosition


class PortfolioSummaryResponse(BaseModel):
    """Response for /api/portfolio-summary — the full report."""

    total_exposure: float
    total_cost_basis: float
    total_unrealized_pnl: float
    position_count: int
    hhi: float
    hhi_classification: str
    effective_positions: float
    largest_position: LargestPosition
    tiers: dict[str, TierMetrics]
    holdings: list[dict]
