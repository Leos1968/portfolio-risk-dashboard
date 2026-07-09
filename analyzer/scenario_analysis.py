"""Historical scenario replay.

Fixes vs the old version:
- `.cumsum()[-1]` crashed with a KeyError: on a date-indexed Series,
  `[-1]` is a *label* lookup (pandas looks for a date equal to -1).
  Positional access must use `.iloc[-1]` — and compounding should be
  multiplicative anyway, so we use (1+r).prod() - 1.
- Empty windows (ticker didn't exist yet, market holiday ranges, Yahoo
  hiccup) now return None instead of crashing the app.
"""

from __future__ import annotations

import numpy as np

from risk_engine import fetch_close_prices, portfolio_return, total_return

# name -> (start, end, what happened)
SCENARIOS = {
    "2008 Financial Crisis": ("2007-10-09", "2009-03-09", "S&P 500 peak-to-trough of the GFC (−57%)"),
    "COVID-19 Crash": ("2020-02-19", "2020-03-23", "Fastest 30%+ drawdown on record"),
    "2022 Rate-Hike Bear": ("2022-01-03", "2022-10-12", "Inflation + rate-shock selloff (−25%)"),
    "2023–24 Bull Run": ("2023-01-03", "2024-12-31", "AI-led recovery rally"),
}


def scenario_performance(tickers: list[str], weights, start, end) -> float | None:
    """Compounded portfolio return (decimal) over [start, end].

    Returns None when no price data exists in the window so callers can
    show a message instead of crashing.
    """
    prices = fetch_close_prices(tickers, start=start, end=end)
    if prices.empty or len(prices) < 2:
        return None
    daily = portfolio_return(prices, np.asarray(weights, dtype=float))
    return total_return(daily)
