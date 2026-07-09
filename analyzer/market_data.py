"""Live market snapshot: major indices, volatility regime, and sector heat.

Everything is fetched through the no-raise `fetch_close_prices`, so a Yahoo
outage degrades to an empty snapshot instead of crashing the page.
"""

from __future__ import annotations

import pandas as pd

from risk_engine import fetch_close_prices

# symbol -> (display name, kind)  — kind drives formatting
INDICES = {
    "^GSPC": ("S&P 500", "index"),
    "^IXIC": ("Nasdaq", "index"),
    "^DJI": ("Dow Jones", "index"),
    "^RUT": ("Russell 2000", "index"),
    "^VIX": ("VIX (fear gauge)", "level"),
    "^TNX": ("10-Yr Treasury", "yield"),  # quoted ×10: 43.5 => 4.35%
    "GC=F": ("Gold", "commodity"),
    "CL=F": ("Crude Oil (WTI)", "commodity"),
}

SECTORS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLY": "Consumer Disc.",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communications",
}


def _last_prev(series: pd.Series) -> tuple[float, float] | None:
    s = series.dropna()
    if len(s) < 2:
        return None
    return float(s.iloc[-1]), float(s.iloc[-2])


def market_snapshot() -> dict:
    """Daily board for the major indices + YTD context.

    Returns {"available": bool, "as_of": str, "rows": [
        {symbol, name, kind, last, day_pct, ytd_pct}, ...]}
    """
    symbols = list(INDICES)
    prices = fetch_close_prices(symbols, period="ytd")
    if prices.empty or len(prices) < 2:
        # YTD can legitimately be 1 row in early January — retry with 6mo
        prices = fetch_close_prices(symbols, period="6mo")
    if prices.empty or len(prices) < 2:
        return {"available": False, "rows": []}

    rows = []
    for sym, (name, kind) in INDICES.items():
        if sym not in prices.columns:
            continue
        lp = _last_prev(prices[sym])
        if lp is None:
            continue
        last, prev = lp
        first = float(prices[sym].dropna().iloc[0])
        rows.append(
            {
                "symbol": sym,
                "name": name,
                "kind": kind,
                "last": last,
                "day_pct": last / prev - 1.0 if prev else 0.0,
                "ytd_pct": last / first - 1.0 if first else 0.0,
            }
        )
    return {
        "available": bool(rows),
        "as_of": str(prices.index[-1].date()),
        "rows": rows,
    }


def vix_regime(vix_level: float) -> tuple[str, str]:
    """Map the VIX to a plain-English market-stress regime."""
    if vix_level < 15:
        return "Calm", "Markets are complacent — hedges are cheap."
    if vix_level < 20:
        return "Normal", "Typical day-to-day volatility."
    if vix_level < 30:
        return "Elevated", "Investors are nervous — expect bigger swings."
    return "Stressed", "Crisis-level fear — historically both danger and opportunity."


def sector_heat() -> dict:
    """Daily and YTD performance for the 11 S&P sector ETFs."""
    prices = fetch_close_prices(list(SECTORS), period="ytd")
    if prices.empty or len(prices) < 2:
        prices = fetch_close_prices(list(SECTORS), period="6mo")
    if prices.empty or len(prices) < 2:
        return {"available": False, "rows": []}

    rows = []
    for sym, name in SECTORS.items():
        if sym not in prices.columns:
            continue
        lp = _last_prev(prices[sym])
        if lp is None:
            continue
        last, prev = lp
        first = float(prices[sym].dropna().iloc[0])
        rows.append(
            {
                "symbol": sym,
                "sector": name,
                "day_pct": last / prev - 1.0 if prev else 0.0,
                "ytd_pct": last / first - 1.0 if first else 0.0,
            }
        )
    rows.sort(key=lambda r: r["day_pct"], reverse=True)
    return {"available": bool(rows), "as_of": str(prices.index[-1].date()), "rows": rows}
