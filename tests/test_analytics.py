"""Offline unit tests for the analytics engine.

Market data is stubbed via monkeypatching `fetch_close_prices`, so the suite
runs without network access and never depends on Yahoo Finance being up.
"""

import numpy as np
import pandas as pd
import pytest

from backend import analytics


@pytest.fixture()
def holdings_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT", "TLT"],
            "Shares": [10, 5, 20],
            "Average_Price": [150.0, 300.0, 100.0],
            "Current_Price": [200.0, 400.0, 90.0],
            "Tier": ["Growth", "Core", "Defensive"],
        }
    )


@pytest.fixture()
def synthetic_prices() -> pd.DataFrame:
    """~1 year of correlated GBM-style prices for assets + benchmark."""
    rng = np.random.default_rng(7)
    n = 252
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    market = rng.normal(0.0004, 0.011, n)
    data = {
        "AAPL": 200 * np.exp(np.cumsum(1.2 * market + rng.normal(0, 0.008, n))),
        "MSFT": 400 * np.exp(np.cumsum(1.0 * market + rng.normal(0, 0.007, n))),
        "TLT": 90 * np.exp(np.cumsum(-0.3 * market + rng.normal(0, 0.006, n))),
        analytics.BENCHMARK: 5000 * np.exp(np.cumsum(market)),
    }
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_missing_column_raises(holdings_df):
    with pytest.raises(ValueError, match="missing required column"):
        analytics.validate_holdings(holdings_df.drop(columns=["Shares"]))


def test_validate_normalizes_ticker_and_tier(holdings_df):
    df = holdings_df.copy()
    df.loc[0, "Ticker"] = " aapl "
    df.loc[0, "Tier"] = "growth"
    df.loc[1, "Tier"] = "SomethingWeird"
    clean = analytics.validate_holdings(df)
    assert clean.loc[0, "Ticker"] == "AAPL"
    assert clean.loc[0, "Tier"] == "Growth"
    assert clean.loc[1, "Tier"] == "Core"  # unknown tiers fold into Core


def test_validate_drops_bad_rows(holdings_df):
    df = holdings_df.copy()
    df["Current_Price"] = df["Current_Price"].astype(object)
    df.loc[0, "Shares"] = -5
    df.loc[1, "Current_Price"] = "not-a-number"
    clean = analytics.validate_holdings(df)
    assert list(clean["Ticker"]) == ["TLT"]


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def test_hhi_single_position_is_max():
    assert analytics.compute_hhi(np.array([1.0])) == pytest.approx(10_000.0)


def test_hhi_equal_weights():
    hhi = analytics.compute_hhi(np.array([0.25] * 4))
    assert hhi == pytest.approx(2_500.0)
    assert analytics.effective_positions(hhi) == pytest.approx(4.0)


def test_exposure_and_weights_sum_to_one(holdings_df):
    exposed = analytics.compute_exposure(analytics.validate_holdings(holdings_df))
    assert exposed["Weight"].sum() == pytest.approx(1.0)
    assert exposed.loc[0, "Exposure"] == pytest.approx(10 * 200.0)


# ---------------------------------------------------------------------------
# Risk metrics with stubbed market data
# ---------------------------------------------------------------------------

def test_full_report_with_live_data(holdings_df, synthetic_prices, monkeypatch):
    monkeypatch.setattr(analytics, "fetch_close_prices", lambda *a, **k: synthetic_prices)
    report = analytics.portfolio_report(holdings_df)

    assert report["market_data"] == "live"
    # Live prices override the CSV Current_Price
    assert report["holdings"][0]["Current_Price"] == pytest.approx(
        synthetic_prices["AAPL"].iloc[-1], rel=1e-3
    )
    assert 0 < report["var_95_percent"] < 20
    assert report["performance"]["available"]
    assert -1 < report["performance"]["max_drawdown"] <= 0
    assert report["correlations"]["available"]
    n = len(report["correlations"]["tickers"])
    assert len(report["correlations"]["matrix"]) == n
    assert len(report["stress_tests"]) == len(analytics.STRESS_SCENARIOS)
    # Stress impacts scale with beta and total value
    crisis = next(s for s in report["stress_tests"] if "2008" in s["scenario"])
    assert crisis["portfolio_impact_dollar"] == pytest.approx(
        report["total_exposure"] * report["portfolio_beta"] * -0.57, abs=0.01
    )


def test_report_degrades_gracefully_without_market_data(holdings_df, monkeypatch):
    monkeypatch.setattr(analytics, "fetch_close_prices", lambda *a, **k: pd.DataFrame())
    report = analytics.portfolio_report(holdings_df)

    assert report["market_data"] == "unavailable"
    assert report["portfolio_beta"] == 1.0
    assert report["performance"] == {"available": False}
    assert report["correlations"] == {"available": False}
    # Falls back to the CSV prices
    assert report["holdings"][0]["Current_Price"] == pytest.approx(200.0)
    # VaR still produced from the fallback volatility
    assert report["var_dollar"] > 0


def test_benchmark_only_download_counts_as_unavailable(holdings_df, synthetic_prices, monkeypatch):
    bench_only = synthetic_prices[[analytics.BENCHMARK]]
    monkeypatch.setattr(analytics, "fetch_close_prices", lambda *a, **k: bench_only)
    report = analytics.portfolio_report(holdings_df)
    assert report["market_data"] == "unavailable"


def test_report_is_json_serializable(holdings_df, synthetic_prices, monkeypatch):
    import json

    monkeypatch.setattr(analytics, "fetch_close_prices", lambda *a, **k: synthetic_prices)
    report = analytics.portfolio_report(holdings_df)
    json.dumps(report)  # raises TypeError on numpy types, NaN handled upstream
