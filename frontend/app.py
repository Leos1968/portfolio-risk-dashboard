"""
Institutional Portfolio Risk & Analytics Dashboard — Streamlit frontend.

Connects to the FastAPI quantitative backend when available; otherwise falls
back to importing the analytics engine directly, so the app is fully
functional standalone (`streamlit run frontend/app.py`) with zero setup.

Data ingestion:
    - Sidebar CSV uploader (schema: Ticker, Shares, Average_Price,
      Current_Price, Tier)
    - Defaults to a pre-loaded synthetic mock portfolio when no file is
      uploaded, so the app works instantly out of the box.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Allow `from backend import analytics` when run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import analytics  # noqa: E402

API_URL = os.getenv("PORTFOLIO_API_URL", "http://localhost:8000")

TIER_COLORS = {"Growth": "#2A6FDB", "Core": "#1F8A5B", "Defensive": "#8A6D1F"}

st.set_page_config(
    page_title="Portfolio Risk & Analytics",
    page_icon="📊",
    layout="wide",
)


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
    st.sidebar.header("Data Input")
    uploaded = st.sidebar.file_uploader(
        "Upload your Holdings CSV (optional)",
        type=["csv"],
        help="Columns: Ticker, Shares, Average_Price, Current_Price, Tier",
    )
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.sidebar.success(f"Loaded {len(df)} rows from {uploaded.name}")
            return df
        except Exception as exc:  # noqa: BLE001 — surface any parse error to the user
            st.sidebar.error(f"Could not parse CSV: {exc}")
    st.sidebar.info("Using default mock portfolio. Upload a CSV to view your own.")
    return default_mock_portfolio()


# ---------------------------------------------------------------------------
# Analytics (API-first, local fallback)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
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
            marker={"colors": [TIER_COLORS.get(t, "#666") for t in holdings["Tier"]]},
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Exposure: $%{value:,.0f}<br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False, height=380)
    return fig


def hhi_gauge(hhi: float, classification: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=hhi,
            number={"valueformat": ",.0f"},
            title={"text": f"HHI — {classification}", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 10_000]},
                "bar": {"color": "#1a1a1a"},
                "steps": [
                    {"range": [0, analytics.HHI_DIVERSIFIED_MAX], "color": "#c9e4d2"},
                    {"range": [analytics.HHI_DIVERSIFIED_MAX, analytics.HHI_MODERATE_MAX], "color": "#f2e2b3"},
                    {"range": [analytics.HHI_MODERATE_MAX, 10_000], "color": "#f0c3bb"},
                ],
            },
        )
    )
    fig.update_layout(margin=dict(t=40, b=10, l=30, r=30), height=380)
    return fig


def tier_growth_chart(trajectories: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for tier in [c for c in trajectories.columns if c != "Period"]:
        fig.add_trace(
            go.Scatter(
                x=trajectories["Period"],
                y=trajectories[tier],
                name=tier,
                mode="lines",
                line={"color": TIER_COLORS.get(tier, "#666"), "width": 2.5},
                hovertemplate=f"<b>{tier}</b><br>%{{x|%b %Y}}: %{{y:.1f}}<extra></extra>",
            )
        )
    fig.add_hline(y=100, line_dash="dot", line_color="#999")
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=380,
        yaxis_title="Indexed value (start = 100)",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Institutional Portfolio Risk & Analytics")

    raw = load_holdings()
    with st.spinner("Fetching live market data and calculating risk metrics..."):
        try:
            report, source = get_report(raw)
        except ValueError as exc:
            st.error(f"Data validation failed: {exc}")
            st.stop()

    st.caption(
        "⚡ Analytics & Live Market Data served by FastAPI backend" if source == "api"
        else "💻 Backend offline — computations & live market data processed locally."
    )

    holdings = pd.DataFrame(report["holdings"])

    # --- KPI row (Upgraded to 6 columns for new metrics) -------------------
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    c1.metric("Total Exposure", f"${report['total_exposure']:,.0f}")
    
    c2.metric(
        "Unrealized PnL",
        f"${report['total_unrealized_pnl']:,.0f}",
        f"{report['total_unrealized_pnl'] / report['total_cost_basis']:+.1%}",
    )
    
    c3.metric("Concentration (HHI)", f"{report['hhi']:,.0f}", report["hhi_classification"], delta_color="off")
    
    largest = report.get("largest_position", {"ticker": "N/A", "weight": 0})
    c4.metric(
        "Largest Position",
        largest["ticker"],
        f"{largest['weight']:.1%} of book",
        delta_color="off",
    )
    
    # Safely extract the new metrics depending on dict keys
    beta_val = report.get("portfolio_beta", report.get("beta", "N/A"))
    if isinstance(beta_val, (int, float)):
        c5.metric("Portfolio Beta (β)", f"{beta_val:.2f}")
    else:
        c5.metric("Portfolio Beta (β)", "N/A")
        
    var_val = report.get("var_dollar", report.get("var_95", "N/A"))
    if isinstance(var_val, (int, float)):
        c6.metric("95% 1-Day VaR", f"${var_val:,.0f}")
    else:
        c6.metric("95% 1-Day VaR", "N/A")

    st.markdown("---")

    # --- Charts ------------------------------------------------------------
    left, right = st.columns(2)
    with left:
        st.subheader("Asset Exposure Distribution")
        st.plotly_chart(exposure_pie(holdings), use_container_width=True)
    with right:
        st.subheader("Concentration Risk")
        st.plotly_chart(hhi_gauge(report["hhi"], report["hhi_classification"]), use_container_width=True)

    st.subheader("Tier Growth Trajectories (trailing 12 months)")
    
    # Handle trajectories dynamically depending on API structure
    if "trajectories" in report:
        traj_df = pd.DataFrame(report["trajectories"])
        traj_df["Period"] = pd.to_datetime(traj_df["Period"])
        st.plotly_chart(tier_growth_chart(traj_df), use_container_width=True)
    else:
        exposed = analytics.compute_exposure(analytics.validate_holdings(raw))
        st.plotly_chart(tier_growth_chart(analytics.tier_growth_trajectories(exposed)), use_container_width=True)

    # --- Tier summary + holdings detail ------------------------------------
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
    st.dataframe(
        holdings[
            ["Ticker", "Tier", "Shares", "Average_Price", "Current_Price",
             "Exposure", "Unrealized_PnL", "Return_Pct", "Weight"]
        ].style.format(
            {
                "Average_Price": "${:,.2f}",
                "Current_Price": "${:,.2f}",
                "Exposure": "${:,.0f}",
                "Unrealized_PnL": "${:,.0f}",
                "Return_Pct": "{:+.1%}",
                "Weight": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()