"""Market Command Center — live Wall Street briefing + portfolio risk lab.

A standalone Streamlit app (deployed separately from the flagship dashboard
in ../frontend). Five tabs:

    📰 Market Briefing  — daily index board, VIX regime, sector heat
    📊 Portfolio Lab    — build & analyze any ticker/weight portfolio
    🎲 Monte Carlo      — simulate thousands of possible years ahead
    🌪️ Scenarios        — replay 2008, COVID, 2022 against your portfolio
    🧭 Learn            — methodology, glossary, and curated resources

All market data comes from Yahoo Finance through a no-raise fetcher and
degrades gracefully when rate-limited. All return math is in decimals and
compounded — see risk_engine.py.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_data import INDICES, market_snapshot, sector_heat, vix_regime
from risk_engine import (
    TRADING_DAYS,
    fetch_close_prices,
    monte_carlo_final_returns,
    parse_weights,
    portfolio_return,
    total_return,
)
from scenario_analysis import SCENARIOS, scenario_performance

GITHUB_URL = "https://github.com/Leos1968/portfolio-risk-dashboard"
AUTHOR_URL = "https://jerieldeleon.netlify.app"
BENCHMARK = "^GSPC"

# Validated dark-surface palette (shared with the flagship app)
BLUE, GOOD, RED, CRITICAL = "#3987e5", "#0ca30c", "#e66767", "#d03b3b"
MUTED, GRID, BASELINE, INK = "#898781", "#2c2c2a", "#383835", "#fafafa"

st.set_page_config(page_title="Market Command Center", page_icon="🛰️", layout="wide")


def _style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(t=30, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="system-ui, 'Segoe UI', sans-serif"),
        hoverlabel=dict(bgcolor="#1a1a19", font_color=INK, bordercolor=BASELINE),
        legend=dict(orientation="h", y=1.1, font=dict(color=MUTED)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BASELINE, tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, tickfont=dict(color=MUTED), zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=900)
def load_market() -> tuple[dict, dict]:
    return market_snapshot(), sector_heat()


@st.cache_data(show_spinner=False, ttl=900)
def load_prices(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    return fetch_close_prices(list(tickers) + [BENCHMARK], start=start)


@st.cache_data(show_spinner=False, ttl=3600)
def load_scenario(tickers: tuple[str, ...], weights: tuple[float, ...], start: str, end: str):
    return scenario_performance(list(tickers), list(weights), start, end)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🛰️ Market Command Center")
st.sidebar.caption(
    "Live Wall Street data · quantitative risk analytics · zero setup. "
    "Data refreshes every 15 minutes."
)
if st.sidebar.button("🔄 Refresh all market data"):
    st.cache_data.clear()
st.sidebar.divider()
st.sidebar.markdown(
    f"**Explore more**\n\n"
    f"[📊 Institutional dashboard (sister app)](https://portfolio-risk-dashboard.streamlit.app)  \n"
    f"[⭐ Source code on GitHub]({GITHUB_URL})  \n"
    f"[👤 Built by Jeriel De Leon]({AUTHOR_URL})"
)
st.sidebar.caption("Educational tool — not investment advice.")

st.title("🛰️ Market Command Center")

tab_brief, tab_lab, tab_mc, tab_scen, tab_learn = st.tabs(
    ["📰 Market Briefing", "📊 Portfolio Lab", "🎲 Monte Carlo", "🌪️ Scenarios", "🧭 Learn"]
)


# ---------------------------------------------------------------------------
# 📰 Market Briefing
# ---------------------------------------------------------------------------

with tab_brief:
    with st.spinner("Pulling today's Wall Street data..."):
        snap, sectors = load_market()

    if not snap["available"]:
        st.warning("Yahoo Finance is rate-limiting right now — the briefing will be back in a few minutes.")
    else:
        st.caption(f"As of market close **{snap['as_of']}** · source: Yahoo Finance")

        rows = {r["symbol"]: r for r in snap["rows"]}
        board = [s for s in INDICES if s in rows]
        for i in range(0, len(board), 4):
            cols = st.columns(4)
            for col, sym in zip(cols, board[i:i + 4]):
                r = rows[sym]
                if r["kind"] == "yield":
                    col.metric(r["name"], f"{r['last'] / 10:.2f}%", f"{r['day_pct']:+.1%} today",
                               delta_color="inverse",
                               help="Higher yields pressure stock valuations")
                elif r["kind"] == "level":
                    col.metric(r["name"], f"{r['last']:.1f}", f"{r['day_pct']:+.1%} today",
                               delta_color="inverse",
                               help="Implied 30-day S&P 500 volatility — the market's fear thermometer")
                else:
                    col.metric(r["name"], f"{r['last']:,.0f}" if r["kind"] == "index" else f"${r['last']:,.1f}",
                               f"{r['day_pct']:+.1%} today",
                               help=f"YTD: {r['ytd_pct']:+.1%}")

        if "^VIX" in rows:
            regime, blurb = vix_regime(rows["^VIX"]["last"])
            icon = {"Calm": "🟢", "Normal": "🔵", "Elevated": "🟠", "Stressed": "🔴"}[regime]
            st.info(f"{icon} **Volatility regime: {regime}** (VIX {rows['^VIX']['last']:.1f}) — {blurb}")

        st.subheader("Sector Performance — today")
        if sectors["available"]:
            srows = sectors["rows"]
            fig = go.Figure(
                go.Bar(
                    x=[r["sector"] for r in srows],
                    y=[r["day_pct"] * 100 for r in srows],
                    marker={"color": [GOOD if r["day_pct"] >= 0 else RED for r in srows],
                            "cornerradius": 4},
                    customdata=[[r["ytd_pct"] * 100] for r in srows],
                    hovertemplate="<b>%{x}</b><br>Today: %{y:+.2f}%<br>YTD: %{customdata[0]:+.1f}%<extra></extra>",
                )
            )
            fig.add_hline(y=0, line_color=BASELINE)
            fig.update_layout(showlegend=False, yaxis_title="Daily change (%)")
            st.plotly_chart(_style(fig, 360), use_container_width=True)

            best, worst = srows[0], srows[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Leading sector", best["sector"], f"{best['day_pct']:+.2%} today", delta_color="off")
            c2.metric("Lagging sector", worst["sector"], f"{worst['day_pct']:+.2%} today", delta_color="off")
            breadth = sum(1 for r in srows if r["day_pct"] > 0)
            c3.metric("Market breadth", f"{breadth} of {len(srows)} sectors up",
                      "risk-on tone" if breadth >= 7 else ("mixed tape" if breadth >= 4 else "risk-off tone"),
                      delta_color="off",
                      help="How many of the 11 S&P sector ETFs are positive today")

        with st.expander("📖 How to read this page"):
            st.markdown(
                "**Indices** show where the broad market is; **the VIX** shows how scared it is; "
                "**sector performance** shows *who* is winning today. Traders read all three together: "
                "e.g. tech leading + low VIX + most sectors green = classic risk-on day. Defensive "
                "sectors (Staples, Utilities) leading while the VIX climbs = investors hiding.\n\n"
                "*10-Yr Treasury yield is quoted from ^TNX (index value ÷ 10). Commodity prices are "
                "front-month futures. Data may be delayed ~15 minutes.*"
            )


# ---------------------------------------------------------------------------
# 📊 Portfolio Lab
# ---------------------------------------------------------------------------

with tab_lab:
    st.subheader("Build your portfolio")
    with st.form("portfolio_form"):
        tickers_raw = st.text_input(
            "Tickers (comma separated):", value="AAPL,MSFT,NVDA,TLT",
            help="Any Yahoo Finance symbols — stocks, ETFs, index funds.",
        )
        weights_raw = st.text_input(
            "Portfolio weights (comma separated) (do not add a percentage sign — e.g. 15,30,45,10):",
            value="25,25,25,25",
            help="Plain numbers only. 15,30,45,10 and 0.15,0.30,0.45,0.10 both work — they're normalized to 100%.",
        )
        start_date = st.date_input(
            "Analysis start date", value=dt.date(2020, 1, 1),
            min_value=dt.date(2000, 1, 3), max_value=dt.date.today() - dt.timedelta(days=30),
        )
        submitted = st.form_submit_button("Analyze Portfolio", type="primary")

    if submitted:
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        weights, err = (None, "Enter at least one ticker.") if not tickers else parse_weights(weights_raw, len(tickers))
        if err:
            st.warning(err)
        else:
            st.session_state.tickers = tickers
            st.session_state.weights = weights.tolist()
            st.session_state.start = str(start_date)

    if "tickers" not in st.session_state:
        st.info("👆 Enter tickers and weights, then click **Analyze Portfolio**. "
                "The Monte Carlo and Scenarios tabs unlock once a portfolio is loaded.")
    else:
        tickers = st.session_state.tickers
        weights = np.array(st.session_state.weights)

        with st.spinner("Fetching market data..."):
            prices_all = load_prices(tuple(tickers), st.session_state.start)

        if prices_all.empty:
            st.error("No price data returned (bad tickers, or Yahoo is rate-limiting). Try again shortly.")
            st.stop()

        missing = [t for t in tickers if t not in prices_all.columns]
        if missing:
            st.warning(f"No data for: {', '.join(missing)} — continuing without them (weights renormalized).")
            kept = [t for t in tickers if t in prices_all.columns]
            keep_w = np.array([w for t, w in zip(tickers, weights) if t in prices_all.columns])
            if not kept:
                st.error("None of the tickers returned data.")
                st.stop()
            tickers, weights = kept, keep_w / keep_w.sum()

        prices = prices_all[tickers]
        daily = portfolio_return(prices, weights)
        st.session_state.daily_returns = daily  # reused by the MC tab

        tot = total_return(daily)
        ann_vol = float(daily.std() * np.sqrt(TRADING_DAYS))
        years = max(len(daily) / TRADING_DAYS, 1e-9)
        cagr = (1.0 + tot) ** (1.0 / years) - 1.0 if tot is not None and tot > -1 else None
        sharpe = (daily.mean() * TRADING_DAYS - 0.045) / ann_vol if ann_vol > 0 else 0.0
        index = 100.0 * (1.0 + daily).cumprod()
        mdd = float((index / index.cummax() - 1.0).min())

        st.subheader("Performance")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Return", f"{tot:+.1%}" if tot is not None else "—",
                  help="Compounded growth since your start date")
        c2.metric("CAGR", f"{cagr:+.1%}" if cagr is not None else "—",
                  help="Compound annual growth rate")
        c3.metric("Annual Volatility", f"{ann_vol:.1%}",
                  help="Annualized standard deviation of daily returns")
        c4.metric("Sharpe Ratio", f"{sharpe:.2f}",
                  help="Risk-adjusted return (4.5% risk-free). Above 1 is good.")
        c5.metric("Max Drawdown", f"{mdd:.1%}",
                  help="Worst peak-to-trough fall — the pain metric")

        # Portfolio vs S&P 500, both indexed to 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=index.index, y=index, name="Your Portfolio", mode="lines",
                                 line={"color": BLUE, "width": 2.5},
                                 hovertemplate="Portfolio: %{y:.1f}<extra></extra>"))
        if BENCHMARK in prices_all.columns:
            bench_daily = prices_all[BENCHMARK].pct_change().dropna()
            bench_index = (100.0 * (1.0 + bench_daily).cumprod()).reindex(index.index).ffill()
            fig.add_trace(go.Scatter(x=index.index, y=bench_index, name="S&P 500", mode="lines",
                                     line={"color": MUTED, "width": 2},
                                     hovertemplate="S&P 500: %{y:.1f}<extra></extra>"))
        fig.add_hline(y=100, line_dash="dot", line_color=BASELINE)
        fig.update_layout(hovermode="x unified", yaxis_title="Growth of 100")
        st.plotly_chart(_style(fig, 420), use_container_width=True)

        # Per-holding statistics
        st.subheader("Holdings breakdown")
        rets = prices.pct_change().dropna(how="all")
        bench_rets = prices_all[BENCHMARK].pct_change().dropna() if BENCHMARK in prices_all.columns else None
        bvar = float(bench_rets.var()) if bench_rets is not None and len(bench_rets) > 1 else None
        rows = []
        for t, w in zip(tickers, weights):
            s = prices[t].dropna()
            r = rets[t].dropna()
            beta = float(r.cov(bench_rets) / bvar) if bvar else None
            rows.append({
                "Ticker": t,
                "Weight": float(w) * 100,
                "Last Price": float(s.iloc[-1]),
                "Day": float(s.iloc[-1] / s.iloc[-2] - 1) * 100 if len(s) > 1 else 0.0,
                "Since Start": float(s.iloc[-1] / s.iloc[0] - 1) * 100,
                "Volatility (ann.)": float(r.std() * np.sqrt(TRADING_DAYS)) * 100,
                "Beta": round(beta, 2) if beta is not None else None,
                "Quote": f"https://finance.yahoo.com/quote/{t}",
            })
        st.dataframe(
            pd.DataFrame(rows), use_container_width=True, hide_index=True,
            column_config={
                "Weight": st.column_config.ProgressColumn("Weight", format="%.1f%%", min_value=0,
                                                          max_value=max(r["Weight"] for r in rows)),
                "Last Price": st.column_config.NumberColumn(format="$%.2f"),
                "Day": st.column_config.NumberColumn("Today", format="%+.2f%%"),
                "Since Start": st.column_config.NumberColumn(format="%+.1f%%"),
                "Volatility (ann.)": st.column_config.NumberColumn("Volatility", format="%.1f%%"),
                "Beta": st.column_config.NumberColumn(help="Sensitivity to S&P 500 moves (1.0 = market-like)"),
                "Quote": st.column_config.LinkColumn("Yahoo", display_text="Open ↗"),
            },
        )


# ---------------------------------------------------------------------------
# 🎲 Monte Carlo
# ---------------------------------------------------------------------------

with tab_mc:
    if "daily_returns" not in st.session_state:
        st.info("Load a portfolio in the **Portfolio Lab** tab first.")
    else:
        daily = st.session_state.daily_returns
        st.subheader("Monte Carlo Simulation")
        st.caption("Thousands of alternate futures, simulated from your portfolio's own history.")

        c1, c2 = st.columns(2)
        horizon_label = c1.selectbox("Horizon", ["3 months", "6 months", "1 year", "2 years"], index=2)
        horizon = {"3 months": 63, "6 months": 126, "1 year": 252, "2 years": 504}[horizon_label]
        n_sims = c2.slider("Simulations", 1000, 20000, 5000, step=1000)

        finals = monte_carlo_final_returns(daily, n_sims=n_sims, horizon_days=horizon)

        var5 = float(np.percentile(finals, 5))
        med = float(np.percentile(finals, 50))
        best = float(np.percentile(finals, 95))
        p_loss = float((finals < 0).mean())
        p_20 = float((finals > 0.20).mean())

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("5% worst case (VaR)", f"{var5:+.1%}",
                  help=f"In 95% of simulations the {horizon_label} ends better than this")
        m2.metric("Median outcome", f"{med:+.1%}")
        m3.metric("5% best case", f"{best:+.1%}")
        m4.metric("Chance of a loss", f"{p_loss:.0%}")
        m5.metric("Chance of +20% or more", f"{p_20:.0%}")

        pct = finals * 100
        fig = go.Figure(go.Histogram(x=pct, nbinsx=50, marker={"color": BLUE},
                                     hovertemplate="Return: %{x:.1f}%<br>Simulations: %{y}<extra></extra>"))
        fig.add_vline(x=var5 * 100, line_color=CRITICAL, line_width=2,
                      annotation_text=f"5% VaR: {var5:+.1%}", annotation_font_color=CRITICAL)
        fig.add_vline(x=med * 100, line_color=MUTED, line_dash="dot",
                      annotation_text=f"median {med:+.1%}", annotation_font_color=MUTED)
        fig.update_layout(showlegend=False,
                          xaxis_title=f"Simulated {horizon_label} return (%)",
                          yaxis_title="Number of simulations")
        st.plotly_chart(_style(fig, 400), use_container_width=True)

        table = pd.DataFrame({
            "Percentile": ["1% (disaster)", "5% (bad year)", "25%", "50% (median)", "75%", "95% (great year)"],
            "Outcome": [f"{np.percentile(finals, q):+.1%}" for q in (1, 5, 25, 50, 75, 95)],
        })
        st.dataframe(table, hide_index=True, use_container_width=True)

        with st.expander("📖 How this works — and what it can't tell you"):
            st.markdown(
                "Each simulation draws a fresh sequence of daily returns from a normal distribution "
                "matched to your portfolio's historical mean and volatility, then **compounds** them "
                "over the horizon. The spread of endings estimates your risk.\n\n"
                "**Limits to respect:** real markets have fat tails (extreme days happen more often "
                "than the normal curve says), volatility clusters, and the future need not resemble "
                "your chosen start window. Treat these as *orders of magnitude*, not predictions — "
                "that's how practitioners use them too."
            )


# ---------------------------------------------------------------------------
# 🌪️ Scenarios
# ---------------------------------------------------------------------------

with tab_scen:
    if "tickers" not in st.session_state:
        st.info("Load a portfolio in the **Portfolio Lab** tab first.")
    else:
        tickers = st.session_state.tickers
        weights = np.array(st.session_state.weights)

        st.subheader("Historical Crisis Replay")
        st.caption("Your exact weights, pushed through the actual price history of past market regimes.")

        if st.button("Run Scenario Analysis", type="primary"):
            cols = st.columns(len(SCENARIOS))
            for col, (name, (s, e, blurb)) in zip(cols, SCENARIOS.items()):
                perf = load_scenario(tuple(tickers), tuple(float(w) for w in weights), s, e)
                if perf is None:
                    col.metric(name, "no data", "ticker too young for this window", delta_color="off")
                else:
                    col.metric(name, f"{perf:+.1%}", blurb, delta_color="off")

        st.subheader("Custom window")
        c1, c2 = st.columns(2)
        cs = c1.date_input("From", value=dt.date(2022, 1, 3), key="cust_start")
        ce = c2.date_input("To", value=dt.date(2022, 10, 12), key="cust_end")
        if st.button("Test this window"):
            if cs >= ce:
                st.warning("The start date must be before the end date.")
            else:
                perf = load_scenario(tuple(tickers), tuple(float(w) for w in weights), str(cs), str(ce))
                if perf is None:
                    st.warning("No price data for this portfolio in that window.")
                else:
                    st.metric(f"{cs} → {ce}", f"{perf:+.1%}")

        with st.expander("📖 Why banks stress test"):
            st.markdown(
                "After 2008, regulators made scenario analysis mandatory for every major bank "
                "(the Fed's CCAR/DFAST programs). The question isn't *\"what usually happens?\"* — "
                "that's VaR — but *\"what happens if the worst year we've seen happens again?\"*\n\n"
                "This replay uses **actual historical prices**, so it captures how *your specific mix* "
                "behaved: a bond-heavy book sailed through 2008 far better than a tech-heavy one, but "
                "2022 punished both — the year diversification broke."
            )


# ---------------------------------------------------------------------------
# 🧭 Learn
# ---------------------------------------------------------------------------

with tab_learn:
    st.subheader("How this app works")
    st.markdown(
        f"Every metric is computed live: the app pulls daily prices from Yahoo Finance "
        f"(15-minute cache), converts them to compounded decimal returns, and runs the "
        f"statistics in pandas/NumPy. No databases, no API keys — "
        f"[read the source on GitHub]({GITHUB_URL}). For the institutional version with "
        f"CSV upload, HHI concentration, correlation heatmaps, and parametric-vs-historical VaR, "
        f"see the [sister dashboard](https://portfolio-risk-dashboard.streamlit.app)."
    )

    st.subheader("Glossary")
    glossary = [
        ("CAGR", "The smoothed annual growth rate: what constant yearly return would produce your total return.",
         "https://www.investopedia.com/terms/c/cagr.asp"),
        ("Volatility", "How violently returns swing, annualized. Two portfolios with the same return can carry very different volatility — the calmer one wins.",
         "https://www.investopedia.com/terms/v/volatility.asp"),
        ("Sharpe Ratio", "Return earned per unit of risk, above the risk-free rate. The most quoted number in asset management; above 1 is good.",
         "https://www.investopedia.com/terms/s/sharperatio.asp"),
        ("Max Drawdown", "The worst peak-to-trough fall you'd have lived through. Measures pain, not just risk.",
         "https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp"),
        ("Beta", "How much a holding amplifies S&P 500 moves. 1 = market-like, 2 = double the swings, negative = moves against the market.",
         "https://www.investopedia.com/terms/b/beta.asp"),
        ("Value at Risk (VaR)", "The loss you'd only exceed in the worst 5% of outcomes. A seatbelt gauge — not a worst-case guarantee.",
         "https://www.investopedia.com/terms/v/var.asp"),
        ("VIX", "The market's 30-day implied volatility for the S&P 500 — the 'fear index'. Under 15 is calm; over 30 is crisis territory.",
         "https://www.investopedia.com/terms/v/vix.asp"),
        ("Monte Carlo", "Simulating thousands of random-but-realistic futures to map the range of outcomes instead of guessing one.",
         "https://www.investopedia.com/terms/m/montecarlosimulation.asp"),
    ]
    for name, plain, link in glossary:
        with st.expander(f"**{name}**"):
            st.markdown(f"{plain}\n\n[Deep dive on Investopedia ↗]({link})")

    st.subheader("Curated resources")
    st.markdown(
        "- [Yahoo Finance](https://finance.yahoo.com) — quotes, filings, and the data behind this app\n"
        "- [FRED](https://fred.stlouisfed.org) — the Fed's free macro database (rates, inflation, employment)\n"
        "- [SEC EDGAR](https://www.sec.gov/edgar/search/) — every public company's actual filings\n"
        "- [Damodaran Online](https://pages.stern.nyu.edu/~adamodar/) — NYU's legendary valuation datasets\n"
        "- [Investopedia](https://www.investopedia.com) — the reference for every term in this app\n"
        "- [Fed stress tests (DFAST)](https://www.federalreserve.gov/supervisionreg/dfa-stress-tests.htm) — how the pros do scenario analysis"
    )

    st.caption(
        f"Built by [Jeriel De Leon]({AUTHOR_URL}) · Python, pandas, NumPy, Streamlit, Plotly · "
        "Market data © Yahoo Finance, may be delayed. Educational tool — not investment advice."
    )
