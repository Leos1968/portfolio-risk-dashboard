"""Portfolio Risk Analyzer — rebuilt.

Fixes vs the old version:
1. Weights input now says "no % sign" — and forgives one anyway.
2. Monte Carlo compounds decimal returns correctly (the old version mixed
   percent and decimal scales, producing results like −3,958%).
3. Scenario analysis uses positional indexing and compounded returns, and
   handles empty data windows instead of raising KeyError.

Run: streamlit run files/dashboard.py
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from risk_engine import (
    TRADING_DAYS,
    fetch_close_prices,
    monte_carlo_final_returns,
    parse_weights,
    portfolio_return,
    total_return,
)
from scenario_analysis import SCENARIOS, scenario_performance

st.set_page_config(page_title="Portfolio Risk Analyzer", page_icon="📈", layout="centered")
st.title("📈 Portfolio Risk Analyzer")


@st.cache_data(show_spinner=False, ttl=900)
def load_prices(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    return fetch_close_prices(list(tickers), start=start)


@st.cache_data(show_spinner=False, ttl=3600)
def load_scenario(tickers: tuple[str, ...], weights: tuple[float, ...], start: str, end: str):
    return scenario_performance(list(tickers), list(weights), start, end)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

tickers_raw = st.text_input(
    "Enter tickers (comma separated):",
    value="AAPL,MSFT,NVDA,TLT",
    help="Any Yahoo Finance symbols — stocks, ETFs, indexes.",
)
weights_raw = st.text_input(
    "Enter portfolio weights (comma separated) (do not add a percentage sign — e.g. 15,30,45,10):",
    value="25,25,25,25",
    help="Plain numbers only. 15,30,45,10 and 0.15,0.30,0.45,0.10 both work; they are normalized to 100%.",
)
start_date = st.date_input(
    "Start Date",
    value=dt.date(2020, 1, 1),
    min_value=dt.date(2000, 1, 3),
    max_value=dt.date.today() - dt.timedelta(days=30),
)

if st.button("Analyze Portfolio", type="primary"):
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if not tickers:
        st.warning("Enter at least one ticker.")
        st.stop()

    weights, err = parse_weights(weights_raw, len(tickers))
    if err:
        st.warning(err)
        st.stop()

    st.session_state.tickers = tickers
    st.session_state.weights = weights.tolist()
    st.session_state.start = str(start_date)

# Everything below renders once a valid portfolio is in session state
if "tickers" not in st.session_state:
    st.info("Enter your portfolio above and click **Analyze Portfolio**.")
    st.stop()

tickers = st.session_state.tickers
weights = np.array(st.session_state.weights)

with st.spinner("Fetching market data..."):
    prices = load_prices(tuple(tickers), st.session_state.start)

if prices.empty:
    st.error("No price data returned (bad tickers, or Yahoo Finance is rate-limiting). Try again in a minute.")
    st.stop()

missing = [t for t in tickers if t not in prices.columns]
if missing:
    st.warning(f"No data for: {', '.join(missing)} — continuing with the rest (weights renormalized).")
    kept = [t for t in tickers if t in prices.columns]
    keep_w = np.array([w for t, w in zip(tickers, weights) if t in prices.columns])
    tickers, weights = kept, keep_w / keep_w.sum()
    prices = prices[kept]

daily = portfolio_return(prices, weights)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

st.subheader("Portfolio Performance")
growth = (1.0 + daily).cumprod() * 100.0
st.line_chart(growth.rename("Growth of 100"))

tot = total_return(daily)
ann_vol = float(daily.std() * np.sqrt(TRADING_DAYS))
years = max(len(daily) / TRADING_DAYS, 1e-9)
cagr = (1.0 + tot) ** (1.0 / years) - 1.0 if tot is not None and tot > -1 else None

c1, c2, c3 = st.columns(3)
c1.metric("Total Return", f"{tot:+.1%}" if tot is not None else "—")
c2.metric("Annualized (CAGR)", f"{cagr:+.1%}" if cagr is not None else "—")
c3.metric("Annual Volatility", f"{ann_vol:.1%}")

# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

st.subheader("Monte Carlo Simulation — next 12 months")
n_sims = st.slider("Number of simulations", 1000, 20000, 5000, step=1000)

finals = monte_carlo_final_returns(daily, n_sims=n_sims, horizon_days=TRADING_DAYS)
finals_pct = finals * 100.0

counts, edges = np.histogram(finals_pct, bins=40)
centers = (edges[:-1] + edges[1:]) / 2.0
st.bar_chart(pd.DataFrame({"Simulated 1-year return (%)": counts},
                          index=pd.Index(np.round(centers, 1), name="return %")))

var5 = float(np.percentile(finals, 5))
med = float(np.percentile(finals, 50))
best = float(np.percentile(finals, 95))
m1, m2, m3 = st.columns(3)
m1.metric("5% worst case (VaR)", f"{var5:+.1%}",
          help="In 95% of simulations the year ends better than this")
m2.metric("Median outcome", f"{med:+.1%}")
m3.metric("5% best case", f"{best:+.1%}")
st.caption(
    "Simulated by compounding 252 random daily returns drawn from your portfolio's "
    "historical mean and volatility. Assumes normal returns — real markets have fatter tails."
)

# ---------------------------------------------------------------------------
# Scenario Analysis
# ---------------------------------------------------------------------------

st.subheader("Scenario Analysis")
st.caption("How would this exact portfolio have performed through past market regimes?")

if st.button("Run Scenario Analysis"):
    for name, (s, e, blurb) in SCENARIOS.items():
        perf = load_scenario(tuple(tickers), tuple(float(w) for w in weights), s, e)
        if perf is None:
            st.warning(f"**{name}** ({s} → {e}): no price data for this portfolio in that window "
                       "(a ticker may not have existed yet).")
        else:
            st.metric(name, f"{perf:+.1%}", blurb, delta_color="off")
