"""
Institutional Portfolio Risk & Analytics Dashboard — Streamlit frontend.

Connects to the FastAPI quantitative backend when available; otherwise falls
back to importing the analytics engine directly, so the app is fully
functional standalone (`streamlit run frontend/app.py`) with zero setup.

Layout: a KPI header plus six tabs —
    Overview · Risk · Performance · Correlations · Stress Lab · Learn
Each analytical section pairs the chart with layered explanations
("plain English" → formula → practitioner caveats) so the app is useful
to readers from beginner to expert.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Allow `from backend import analytics` when run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import analytics  # noqa: E402

API_URL = os.getenv("PORTFOLIO_API_URL", "http://localhost:8000")
GITHUB_URL = "https://github.com/Leos1968/portfolio-risk-dashboard"
AUTHOR_URL = "https://jerieldeleon.netlify.app"

# ---------------------------------------------------------------------------
# Chart palette (validated categorical/diverging set for the dark surface)
# ---------------------------------------------------------------------------

BLUE = "#3987e5"      # categorical slot 1 — portfolio / primary series
AQUA = "#199e70"      # slot 2
YELLOW = "#c98500"    # slot 3
RED = "#e66767"       # diverging warm pole / drawdown
CRITICAL = "#d03b3b"  # status: critical (VaR threshold)
GOOD = "#0ca30c"      # status: good
MUTED = "#898781"     # axis labels, benchmark line
GRID = "#2c2c2a"
BASELINE = "#383835"
INK = "#fafafa"

TIER_COLORS = {"Growth": BLUE, "Core": AQUA, "Defensive": YELLOW}

st.set_page_config(
    page_title="Portfolio Risk & Analytics",
    page_icon="📊",
    layout="wide",
)


def _style(fig: go.Figure, height: int = 380) -> go.Figure:
    """Shared chart chrome: transparent surface, recessive grid, muted axes."""
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
# Data ingestion
# ---------------------------------------------------------------------------

def default_mock_portfolio() -> pd.DataFrame:
    """Synthetic institutional-style book used when no CSV is uploaded."""
    return pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT", "NVDA", "O", "JPM", "AVGO", "PG", "XOM", "UNH", "TLT"],
            "Shares": [50, 30, 40, 100, 25, 12, 60, 45, 15, 80],
            "Average_Price": [175.00, 350.00, 450.00, 55.00, 160.00, 900.00, 145.00, 105.00, 480.00, 92.00],
            "Current_Price": [210.00, 420.00, 480.00, 60.00, 195.00, 1150.00, 158.00, 118.00, 510.00, 95.00],
            "Tier": ["Growth", "Core", "Growth", "Defensive", "Core",
                     "Growth", "Defensive", "Core", "Core", "Defensive"],
        }
    )


def load_holdings() -> pd.DataFrame:
    """Sidebar data-input workflow: CSV upload with mock-data fallback."""
    st.sidebar.header("📥 Data Input")
    uploaded = st.sidebar.file_uploader(
        "Upload your Holdings CSV",
        type=["csv"],
        help="Columns: Ticker, Shares, Average_Price, Current_Price, Tier",
    )
    st.sidebar.download_button(
        "⬇️ Download sample CSV",
        default_mock_portfolio().to_csv(index=False).encode(),
        file_name="sample_holdings.csv",
        mime="text/csv",
        help="Use this as a template for your own portfolio",
    )
    if st.sidebar.button("🔄 Refresh market data", help="Clear the cache and re-pull live prices"):
        st.cache_data.clear()

    st.sidebar.divider()
    st.sidebar.markdown(
        f"**About this project**\n\n"
        f"Live risk engine: Python · pandas · FastAPI · Streamlit · Yahoo Finance\n\n"
        f"[🛰️ Market Command Center (sister app)](https://institutional-risk-dashboard.streamlit.app)  \n"
        f"[⭐ Source on GitHub]({GITHUB_URL})  \n"
        f"[👤 Built by Jeriel De Leon]({AUTHOR_URL})"
    )

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.sidebar.success(f"Loaded {len(df)} rows from {uploaded.name}")
            return df
        except Exception as exc:  # noqa: BLE001 — surface any parse error to the user
            st.sidebar.error(f"Could not parse CSV: {exc}")
    st.sidebar.info("Using the demo portfolio. Upload a CSV to analyze your own.")
    return default_mock_portfolio()


# ---------------------------------------------------------------------------
# Analytics (API-first, local fallback)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=900)  # refresh market data every 15 min
def get_report(df: pd.DataFrame) -> tuple[dict, str]:
    """Fetch the analytics report, preferring the FastAPI backend.

    Returns (report, source) where source is "api" or "local".
    """
    payload = {
        "holdings": [
            {
                "ticker": r.Ticker,
                "shares": float(r.Shares),
                "average_price": float(r.Average_Price),
                "current_price": float(r.Current_Price),
                "tier": r.Tier,
            }
            for r in analytics.validate_holdings(df).itertuples()
        ]
    }
    try:
        resp = requests.post(f"{API_URL}/api/portfolio-summary", json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json(), "api"
    except requests.RequestException:
        # Fallback to local engine (which will fetch live yfinance data directly)
        return analytics.portfolio_report(df), "local"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def exposure_pie(holdings: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=holdings["Ticker"],
            values=holdings["Exposure"],
            hole=0.45,
            marker={
                "colors": [TIER_COLORS.get(t, MUTED) for t in holdings["Tier"]],
                "line": {"color": "#0d0d0d", "width": 2},
            },
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Exposure: $%{value:,.0f}<br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    return _style(fig)


def hhi_gauge(hhi: float, classification: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=hhi,
            number={"valueformat": ",.0f", "font": {"color": INK}},
            title={"text": f"HHI — {classification}", "font": {"size": 16, "color": MUTED}},
            gauge={
                "axis": {"range": [0, 10_000], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
                "bar": {"color": BLUE},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, analytics.HHI_DIVERSIFIED_MAX], "color": "#12331f"},
                    {"range": [analytics.HHI_DIVERSIFIED_MAX, analytics.HHI_MODERATE_MAX], "color": "#3a2f10"},
                    {"range": [analytics.HHI_MODERATE_MAX, 10_000], "color": "#3a1512"},
                ],
            },
        )
    )
    return _style(fig)


def performance_chart(perf: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=perf["dates"], y=perf["portfolio_index"],
            name="Your Portfolio", mode="lines",
            line={"color": BLUE, "width": 2.5},
            hovertemplate="Portfolio: %{y:.1f}<extra></extra>",
        )
    )
    if "benchmark_index" in perf:
        fig.add_trace(
            go.Scatter(
                x=perf["dates"], y=perf["benchmark_index"],
                name="S&P 500", mode="lines",
                line={"color": MUTED, "width": 2},
                hovertemplate="S&P 500: %{y:.1f}<extra></extra>",
            )
        )
    fig.add_hline(y=100, line_dash="dot", line_color=BASELINE)
    fig.update_layout(hovermode="x unified", yaxis_title="Growth of 100 (1 year)")
    return _style(fig, height=420)


def drawdown_chart(perf: dict) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=perf["dates"], y=[d * 100 for d in perf["drawdown"]],
            mode="lines", fill="tozeroy",
            line={"color": RED, "width": 1.5},
            fillcolor="rgba(230,103,103,0.25)",
            hovertemplate="Drawdown: %{y:.1f}%<extra></extra>",
            name="Drawdown",
        )
    )
    fig.update_layout(showlegend=False, yaxis_title="Drawdown from peak (%)")
    return _style(fig, height=300)


def beta_bars(individual_betas: dict[str, float]) -> go.Figure:
    items = sorted(individual_betas.items(), key=lambda kv: kv[1], reverse=True)
    tickers = [k for k, _ in items]
    betas = [v for _, v in items]
    fig = go.Figure(
        go.Bar(
            x=tickers, y=betas,
            marker={"color": BLUE, "cornerradius": 4},
            hovertemplate="<b>%{x}</b><br>Beta: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=1.0, line_dash="dot", line_color=MUTED,
        annotation_text="Market (β = 1)", annotation_font_color=MUTED,
    )
    fig.update_layout(showlegend=False, yaxis_title="Beta vs S&P 500")
    return _style(fig, height=360)


def returns_histogram(daily_returns: list[float], var_pct: float, conf: int) -> go.Figure:
    pct = [r * 100 for r in daily_returns]
    fig = go.Figure(
        go.Histogram(
            x=pct, nbinsx=40,
            marker={"color": BLUE},
            hovertemplate="Daily return: %{x:.2f}%<br>Days: %{y}<extra></extra>",
            name="Daily returns",
        )
    )
    fig.add_vline(
        x=-var_pct, line_color=CRITICAL, line_width=2,
        annotation_text=f"{conf}% VaR: −{var_pct:.2f}%",
        annotation_font_color=CRITICAL,
    )
    fig.update_layout(showlegend=False, xaxis_title="Daily return (%)", yaxis_title="Number of days")
    return _style(fig, height=340)


def correlation_heatmap(corr: dict) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=corr["matrix"], x=corr["tickers"], y=corr["tickers"],
            zmin=-1, zmax=1,
            colorscale=[[0.0, BLUE], [0.5, BASELINE], [1.0, RED]],
            text=corr["matrix"],
            texttemplate="%{text:.2f}",
            textfont={"size": 11, "color": INK},
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
            colorbar={"tickfont": {"color": MUTED}, "title": {"text": "ρ", "font": {"color": MUTED}}},
        )
    )
    return _style(fig, height=max(420, 42 * len(corr["tickers"])))


def price_chart(dates: list[str], prices: list[float], ticker: str) -> go.Figure:
    up = prices[-1] >= prices[0]
    fig = go.Figure(
        go.Scatter(
            x=dates, y=prices, mode="lines",
            line={"color": GOOD if up else RED, "width": 2},
            hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>",
            name=ticker,
        )
    )
    fig.update_layout(showlegend=False, yaxis_title=f"{ticker} close ($)")
    return _style(fig, height=340)


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def render_overview(report: dict, holdings: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Asset Exposure")
        st.caption("Colored by tier: 🔵 Growth · 🟢 Core · 🟡 Defensive")
        st.plotly_chart(exposure_pie(holdings), use_container_width=True)
    with right:
        st.subheader("Concentration Risk")
        st.plotly_chart(hhi_gauge(report["hhi"], report["hhi_classification"]), use_container_width=True)
        st.caption(
            f"Effective independent positions: **{report['effective_positions']:.1f}** · "
            f"Largest position: **{report['largest_position']['ticker']}** "
            f"({report['largest_position']['weight']:.1%} of book)"
        )

    with st.expander("📖 What am I looking at?"):
        st.markdown(
            "**Beginner:** The donut shows how your money is split across holdings. The gauge measures "
            "concentration — whether you're dangerously dependent on a few positions.\n\n"
            "**The math:** The Herfindahl-Hirschman Index (HHI) is the sum of squared position weights "
            "(in %), so it ranges from ~0 (infinitely diversified) to 10,000 (one single position). "
            "Antitrust regulators use the same statistic to measure market concentration.\n\n"
            "**Practitioner note:** HHI treats every holding as independent — 10 tech stocks score as "
            "'diversified' even though they crash together. Cross-check with the **Correlations** tab."
        )
        st.latex(r"HHI = \sum_{i=1}^{N} (100 \cdot w_i)^2 \qquad N_{eff} = \frac{10{,}000}{HHI}")

    st.subheader("Tier Summary")
    tiers_df = pd.DataFrame(report["tiers"]).T
    tiers_df.index.name = "Tier"
    st.dataframe(
        tiers_df.style.format(
            {"exposure": "${:,.0f}", "weight": "{:.1%}", "weighted_return": "{:+.1%}", "positions": "{:.0f}"}
        ),
        use_container_width=True,
    )

    st.subheader("Holdings Detail")
    detail = holdings[
        ["Ticker", "Tier", "Shares", "Average_Price", "Current_Price",
         "Exposure", "Unrealized_PnL", "Return_Pct", "Weight"]
    ].copy()
    detail["Weight"] = detail["Weight"] * 100
    detail["Return_Pct"] = detail["Return_Pct"] * 100
    detail["Quote"] = "https://finance.yahoo.com/quote/" + detail["Ticker"]
    st.dataframe(
        detail,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Average_Price": st.column_config.NumberColumn("Avg Cost", format="$%.2f"),
            "Current_Price": st.column_config.NumberColumn("Live Price", format="$%.2f"),
            "Exposure": st.column_config.NumberColumn(format="$%.0f"),
            "Unrealized_PnL": st.column_config.NumberColumn("Unrealized PnL", format="$%.0f"),
            "Return_Pct": st.column_config.NumberColumn("Return", format="%.1f%%"),
            "Weight": st.column_config.ProgressColumn("Weight", format="%.1f%%", min_value=0, max_value=float(detail["Weight"].max())),
            "Quote": st.column_config.LinkColumn("Yahoo Finance", display_text="Open ↗"),
        },
    )


def render_risk(report: dict) -> None:
    total = report["total_exposure"]
    daily_vol = report.get("daily_volatility", 0.02)
    perf = report.get("performance", {})

    st.subheader("Value at Risk — interactive")
    st.caption("Drag the slider: how confident do you want to be about tomorrow's worst case?")
    conf = st.select_slider("Confidence level", options=[90, 95, 99], value=95)
    z_map = {90: 1.282, 95: 1.645, 99: 2.326}
    para_var_pct = z_map[conf] * daily_vol * 100
    para_var_usd = total * z_map[conf] * daily_vol

    hist_var_pct = None
    if perf.get("available") and perf.get("daily_returns"):
        hist_var_pct = -float(np.percentile(perf["daily_returns"], 100 - conf)) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric(
        f"Parametric {conf}% 1-Day VaR", f"${para_var_usd:,.0f}",
        f"−{para_var_pct:.2f}% of book", delta_color="inverse",
        help="Assumes returns are normally distributed: VaR = z × σ_daily × portfolio value",
    )
    if hist_var_pct is not None:
        c2.metric(
            f"Historical {conf}% 1-Day VaR", f"${total * hist_var_pct / 100:,.0f}",
            f"−{hist_var_pct:.2f}% of book", delta_color="inverse",
            help="Model-free: the actual worst daily losses in the past year at this percentile",
        )
        gap = hist_var_pct - para_var_pct
        c3.metric(
            "Fat-tail gap", f"{gap:+.2f} pts",
            "historical vs parametric", delta_color="off",
            help="When historical VaR exceeds parametric VaR, real markets have fatter tails than the normal distribution assumes",
        )

    if perf.get("available") and perf.get("daily_returns"):
        st.plotly_chart(
            returns_histogram(perf["daily_returns"], hist_var_pct if hist_var_pct is not None else para_var_pct, conf),
            use_container_width=True,
        )

    with st.expander("📖 VaR from beginner to expert"):
        st.markdown(
            "**Beginner:** 95% VaR answers: *\"On a normal bad day — the worst 1-in-20 day — how much "
            "could I lose?\"* It is a seatbelt gauge, not a worst-case guarantee.\n\n"
            "**The math:** Parametric VaR assumes daily returns follow a normal distribution with the "
            "portfolio's measured volatility. Historical VaR skips the assumption and reads the loss "
            "percentile straight from the last year of actual returns.\n\n"
            "**Practitioner note:** Real returns have *fat tails* — extreme days happen far more often "
            "than the normal curve predicts (see the gap metric above). That failure mode is exactly why "
            "banks complement VaR with stress testing (next tab) and Expected Shortfall. VaR also says "
            "nothing about *how much worse* than the threshold a loss can get."
        )
        st.latex(r"VaR_{95\%} = 1.645 \cdot \sigma_{daily} \cdot V \qquad VaR^{hist}_{95\%} = -P_{5}(r_1, \dots, r_{252}) \cdot V")

    st.subheader("Systematic Risk — Beta by Holding")
    st.plotly_chart(beta_bars(report["individual_betas"]), use_container_width=True)
    st.caption(
        f"Portfolio beta: **{report['portfolio_beta']:.2f}** — a 1% S&P 500 move implies roughly a "
        f"{report['portfolio_beta']:.2f}% move in this portfolio."
    )
    with st.expander("📖 What is beta?"):
        st.markdown(
            "**Beginner:** Beta measures how much a stock amplifies market moves. β = 1 moves with the "
            "market; β = 2 doubles every market move (both ways); β = 0.5 dampens it; negative β moves "
            "*against* the market (rare — long-duration bonds like TLT often qualify).\n\n"
            "**The math:** β = Cov(asset, market) / Var(market), estimated here on one year of daily "
            "returns against the S&P 500 (^GSPC).\n\n"
            "**Practitioner note:** Beta is backward-looking and unstable across regimes. Low-beta "
            "portfolios can still carry huge idiosyncratic risk — beta only prices the *systematic* part."
        )
        st.latex(r"\beta_i = \frac{\mathrm{Cov}(r_i, r_m)}{\mathrm{Var}(r_m)} \qquad \beta_p = \sum_i w_i \beta_i")


def render_performance(report: dict) -> None:
    perf = report.get("performance", {})
    if not perf.get("available"):
        st.info("Live market data is unavailable right now, so trailing performance can't be computed. Try refreshing in a few minutes.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("1-Year Return", f"{perf['period_return']:+.1%}",
              help="Growth of the current portfolio weights over the trailing year")
    bench = perf.get("benchmark_return")
    if bench is not None:
        c2.metric("vs S&P 500", f"{perf['period_return'] - bench:+.1%}",
                  f"S&P 500: {bench:+.1%}", delta_color="off",
                  help="Excess return over the index — the number active managers are judged on")
    c3.metric("Sharpe Ratio", f"{perf['sharpe_ratio']:.2f}",
              help=f"Excess return per unit of risk, using a {report.get('risk_free_rate', 0.045):.1%} risk-free rate. Above 1 is good, above 2 is excellent.")
    c4.metric("Annual Volatility", f"{perf['annual_volatility']:.1%}",
              help="Standard deviation of daily returns, annualized (×√252)")
    c5.metric("Max Drawdown", f"{perf['max_drawdown']:.1%}",
              help="Worst peak-to-trough decline over the year — the pain metric")

    st.subheader("Portfolio vs S&P 500 — trailing 12 months")
    st.caption("Both series indexed to 100. Hover for daily values; drag to zoom, double-click to reset.")
    st.plotly_chart(performance_chart(perf), use_container_width=True)

    st.subheader("Drawdown Profile")
    st.plotly_chart(drawdown_chart(perf), use_container_width=True)

    with st.expander("📖 Reading these numbers"):
        st.markdown(
            "**Beginner:** The top chart answers *\"did this portfolio beat simply buying the index?\"* "
            "The red chart shows every losing stretch: how deep the portfolio fell from its best point.\n\n"
            "**The math:** Sharpe = (annualized return − risk-free rate) / annualized volatility. "
            "Sortino is the same but only penalizes *downside* volatility "
            f"(here: **{perf['sortino_ratio']:.2f}**).\n\n"
            "**Practitioner note:** This backtest applies *today's* weights to the past year "
            "(no rebalancing, dividends excluded) — a standard simplification; a production system would "
            "use position history and total-return series. Sharpe also treats upside and downside "
            "volatility identically, which is why Sortino exists."
        )
        st.latex(r"Sharpe = \frac{\bar{r}_{ann} - r_f}{\sigma_{ann}} \qquad MDD = \min_t \left( \frac{V_t}{\max_{s \le t} V_s} - 1 \right)")

    st.subheader("🔎 Single-Holding Drill-Down")
    prices = report.get("price_history", {})
    if prices.get("available"):
        tick = st.selectbox("Pick a holding", options=list(prices["series"].keys()))
        series = prices["series"][tick]
        hcol1, hcol2, hcol3, hcol4 = st.columns(4)
        chg = series[-1] / series[0] - 1 if series[0] else 0
        hcol1.metric("Live Price", f"${series[-1]:,.2f}")
        hcol2.metric("1-Year Change", f"{chg:+.1%}")
        hcol3.metric("Beta", f"{report['individual_betas'].get(tick, 1.0):.2f}")
        hcol4.link_button(f"{tick} on Yahoo ↗", f"https://finance.yahoo.com/quote/{tick}")
        st.plotly_chart(price_chart(prices["dates"], series, tick), use_container_width=True)


def render_correlations(report: dict) -> None:
    corr = report.get("correlations", {})
    if not corr.get("available"):
        st.info("Correlation analysis needs live market data for at least two holdings. Try refreshing in a few minutes.")
        return

    st.subheader("Correlation Matrix — daily returns, 1 year")
    st.caption("🔴 +1 = move together · ⚪ 0 = unrelated · 🔵 −1 = move opposite")
    st.plotly_chart(correlation_heatmap(corr), use_container_width=True)

    # Diversification insights: best and worst pairs
    tickers, matrix = corr["tickers"], np.array(corr["matrix"])
    iu = np.triu_indices(len(tickers), k=1)
    if len(iu[0]) > 0:
        pairs = [(tickers[i], tickers[j], matrix[i, j]) for i, j in zip(*iu)]
        hi = max(pairs, key=lambda p: p[2])
        lo = min(pairs, key=lambda p: p[2])
        avg = float(np.mean([p[2] for p in pairs]))
        c1, c2, c3 = st.columns(3)
        c1.metric("Most correlated pair", f"{hi[0]} · {hi[1]}", f"ρ = {hi[2]:+.2f}", delta_color="off",
                  help="These two positions largely duplicate each other's risk")
        c2.metric("Best diversifier pair", f"{lo[0]} · {lo[1]}", f"ρ = {lo[2]:+.2f}", delta_color="off",
                  help="The pair most likely to offset each other in a selloff")
        c3.metric("Average pairwise ρ", f"{avg:+.2f}",
                  help="Lower average correlation = more genuine diversification")

    with st.expander("📖 Why correlation is the whole game"):
        st.markdown(
            "**Beginner:** Diversification only works if your holdings *don't* move together. Ten stocks "
            "that all drop on the same news are effectively one big position.\n\n"
            "**The math:** Correlation (ρ) rescales covariance to [−1, +1]. Portfolio variance is built "
            "from every pairwise covariance — with many holdings, the *covariances* dominate, not the "
            "individual volatilities. This is the engine inside the VaR number on the Risk tab.\n\n"
            "**Practitioner note:** Correlations are regime-dependent and spike toward +1 in crises "
            "(\"diversification fails exactly when you need it\") — a key reason stress tests exist. "
            "Bonds (TLT) historically offer negative equity correlation, but 2022 broke that assumption."
        )
        st.latex(r"\sigma_p^2 = \sum_i \sum_j w_i w_j \, \sigma_i \sigma_j \, \rho_{ij}")


def render_stress(report: dict) -> None:
    total = report["total_exposure"]
    beta = report["portfolio_beta"]

    st.subheader("Historical Crisis Replay")
    st.caption(
        f"First-order estimate: portfolio move ≈ β ({beta:.2f}) × market move. "
        "Each card shows what that crisis would do to this portfolio *today*."
    )
    scenarios = report.get("stress_tests", [])
    cols = st.columns(len(scenarios)) if scenarios else []
    for col, sc in zip(cols, scenarios):
        col.metric(
            sc["scenario"],
            f"${sc['portfolio_impact_dollar']:,.0f}",
            f"{sc['portfolio_impact_pct']:.1%} (mkt {sc['market_move']:+.0%})",
            delta_color="inverse" if sc["portfolio_impact_pct"] < 0 else "normal",
            help=sc["description"],
        )

    st.subheader("🎛️ Build Your Own Scenario")
    shock = st.slider(
        "S&P 500 moves by…", min_value=-60, max_value=20, value=-20, step=5,
        format="%d%%", help="Drag to simulate any market move",
    )
    impact_pct = beta * shock / 100
    impact_usd = total * impact_pct
    c1, c2 = st.columns(2)
    c1.metric("Estimated portfolio impact", f"{impact_pct:+.1%}",
              f"${impact_usd:,.0f}", delta_color="inverse" if impact_usd < 0 else "normal")
    c2.metric("Portfolio value after shock", f"${total + impact_usd:,.0f}",
              f"from ${total:,.0f}", delta_color="off")

    with st.expander("📖 How stress testing works (and its limits)"):
        st.markdown(
            "**Beginner:** Instead of asking *\"what usually happens?\"* (VaR), stress tests ask "
            "*\"what happens if 2008 repeats tomorrow?\"* Regulators require every major bank to run "
            "these (the Fed's CCAR/DFAST programs).\n\n"
            "**The math:** This app uses beta-scaling — the simplest stress model: portfolio shock = "
            "β × market shock.\n\n"
            "**Practitioner note:** Beta-scaling is deliberately first-order. It ignores correlation "
            "breakdown, liquidity spirals, and convexity — real crisis losses usually *exceed* the "
            "beta-scaled estimate for risky books, while flight-to-quality can flip the sign for "
            "defensive assets. Bank stress frameworks shock every risk factor path, not just the index."
        )


def render_learn(report: dict) -> None:
    st.subheader("🎓 How this app works")
    st.markdown(
        f"Every time this page loads, the engine pulls **one year of daily Wall Street prices** for "
        f"every holding plus the S&P 500 from Yahoo Finance, then recomputes every metric you see — "
        f"exposure, PnL, HHI concentration, beta, parametric & historical VaR, Sharpe, drawdown, "
        f"correlations, and stress scenarios. Data refreshes at most every 15 minutes.\n\n"
        f"**Stack:** Python · pandas · NumPy · yfinance · FastAPI (REST API) · Streamlit · Plotly — "
        f"[read the source on GitHub]({GITHUB_URL}).\n\n"
        f"Currently using the **{report.get('market_data', 'live')}** market-data path with "
        f"{report.get('position_count', 0)} positions."
    )

    st.subheader("📚 Glossary — every metric explained")
    glossary = [
        ("Exposure & Weight", "How much money is in each position (shares × live price), and its share of the total book.",
         r"E_i = q_i \cdot p_i \qquad w_i = E_i / \textstyle\sum_j E_j",
         "https://www.investopedia.com/terms/m/marketexposure.asp"),
        ("HHI — Concentration", "Sum of squared weights. The same statistic antitrust regulators use for market concentration, applied to your book. Under 1,500 = diversified; over 2,500 = concentrated.",
         r"HHI = \sum_i (100 w_i)^2",
         "https://www.investopedia.com/terms/h/hhi.asp"),
        ("Beta — Systematic Risk", "How much your portfolio amplifies S&P 500 moves. Estimated from one year of daily returns.",
         r"\beta = \mathrm{Cov}(r_p, r_m)/\mathrm{Var}(r_m)",
         "https://www.investopedia.com/terms/b/beta.asp"),
        ("Value at Risk (VaR)", "The loss threshold you'd only breach on the worst 1-in-20 days (95%). Computed two ways here: parametric (normal-distribution) and historical (actual past losses).",
         r"VaR_{c} = z_c \cdot \sigma \cdot V",
         "https://www.investopedia.com/terms/v/var.asp"),
        ("Sharpe Ratio", "Return earned per unit of risk taken, above the risk-free rate. The single most quoted performance statistic in finance.",
         r"S = (\bar{r} - r_f)/\sigma",
         "https://www.investopedia.com/terms/s/sharperatio.asp"),
        ("Max Drawdown", "The worst peak-to-trough fall. Measures the pain an investor actually had to sit through.",
         r"MDD = \min_t (V_t / \max_{s\le t} V_s - 1)",
         "https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp"),
        ("Correlation", "Whether two holdings move together (+1), independently (0), or opposite (−1). The foundation of diversification.",
         r"\rho_{ij} = \mathrm{Cov}(r_i, r_j)/(\sigma_i \sigma_j)",
         "https://www.investopedia.com/terms/c/correlation.asp"),
        ("Stress Testing", "Replaying historical crises against today's portfolio. Required practice at every major bank since 2008.",
         r"\Delta V \approx \beta \cdot \Delta m \cdot V",
         "https://www.investopedia.com/terms/s/stresstesting.asp"),
    ]
    for name, plain, formula, link in glossary:
        with st.expander(f"**{name}**"):
            st.markdown(plain)
            st.latex(formula)
            st.markdown(f"[Deep dive on Investopedia ↗]({link})")

    st.subheader("🧪 Analyze your own portfolio")
    st.markdown(
        "1. Download the sample CSV from the sidebar\n"
        "2. Replace the rows with your holdings — columns: `Ticker, Shares, Average_Price, Current_Price, Tier` "
        "(tier is your own label: Growth / Core / Defensive)\n"
        "3. Upload it — every tab recomputes instantly with live prices\n\n"
        "*Current_Price is only a fallback; the engine overwrites it with the live market price when available.*"
    )
    st.caption(
        "Educational tool — not investment advice. Market data: Yahoo Finance (may be delayed). "
        f"Built by [Jeriel De Leon]({AUTHOR_URL})."
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📊 Institutional Portfolio Risk & Analytics")

    raw = load_holdings()
    with st.spinner("Fetching live market data and calculating risk metrics..."):
        try:
            report, source = get_report(raw)
        except ValueError as exc:
            st.error(f"Data validation failed: {exc}")
            st.stop()

    st.caption(
        "⚡ Analytics & live market data served by the FastAPI backend" if source == "api"
        else "💻 Live market data pulled directly from Yahoo Finance · auto-refreshes every 15 min"
    )
    if report.get("market_data") == "unavailable":
        st.warning(
            "Live market data is temporarily unavailable (Yahoo Finance rate limit or "
            "network issue). Showing prices from your holdings file; Beta/VaR use "
            "conservative defaults. Refresh in a few minutes."
        )

    holdings = pd.DataFrame(report["holdings"])
    perf = report.get("performance", {})

    # --- KPI row -----------------------------------------------------------
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Exposure", f"${report['total_exposure']:,.0f}",
              help="Market value of all positions at live prices")
    c2.metric(
        "Unrealized PnL",
        f"${report['total_unrealized_pnl']:,.0f}",
        f"{report['total_unrealized_pnl'] / report['total_cost_basis']:+.1%}",
        help="Gain/loss vs what you paid (cost basis)",
    )
    c3.metric("Portfolio Beta (β)", f"{report.get('portfolio_beta', 1.0):.2f}",
              help="Sensitivity to S&P 500 moves — 1.00 means market-like risk")
    c4.metric("95% 1-Day VaR", f"${report.get('var_dollar', 0):,.0f}",
              help="Worst expected 1-day loss at 95% confidence (parametric)")
    if perf.get("available"):
        c5.metric("Sharpe Ratio", f"{perf['sharpe_ratio']:.2f}",
                  help="Risk-adjusted return over the trailing year — above 1 is good")
        c6.metric("Max Drawdown", f"{perf['max_drawdown']:.1%}",
                  help="Worst peak-to-trough decline in the trailing year")
    else:
        c5.metric("Sharpe Ratio", "—")
        c6.metric("Max Drawdown", "—")

    st.markdown("---")

    tabs = st.tabs(["📊 Overview", "⚠️ Risk", "📈 Performance", "🔗 Correlations", "🧪 Stress Lab", "🎓 Learn"])
    with tabs[0]:
        render_overview(report, holdings)
    with tabs[1]:
        render_risk(report)
    with tabs[2]:
        render_performance(report)
    with tabs[3]:
        render_correlations(report)
    with tabs[4]:
        render_stress(report)
    with tabs[5]:
        render_learn(report)


if __name__ == "__main__":
    main()
