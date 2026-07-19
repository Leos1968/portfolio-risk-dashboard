# 📊 Institutional Portfolio Risk & Analytics Dashboard

[![CI](https://github.com/Leos1968/portfolio-risk-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/Leos1968/portfolio-risk-dashboard/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688)

A full-stack quantitative risk engine that pulls **live daily market data from Wall Street** (Yahoo Finance)
and computes the risk metrics used on institutional trading desks — for any portfolio you upload as a CSV.

**▶ Live demos:**
- **[Institutional Risk Dashboard](https://institutional-risk-dashboard.streamlit.app)** — upload a holdings CSV, get a full risk report (`frontend/app.py`)
- **[Market Command Center](https://market-command-dashboard.streamlit.app)** — daily Wall Street briefing + interactive portfolio lab (`analyzer/dashboard.py`)
- **[IB/PE Modeling Suite](https://market-command-dashboard.streamlit.app)** — three-statement · M&A accretion/dilution · LBO returns, a second page of the Market Command Center (`modeling/`)

## What it computes

| Metric | Method |
|---|---|
| **Portfolio Beta (β)** | Covariance of daily returns vs the S&P 500 (^GSPC), 1-year window |
| **Value at Risk — parametric** | `VaR = z · σ_daily · V` from the full asset covariance matrix |
| **Value at Risk — historical** | Loss percentile of actual trailing-1y portfolio returns (model-free) |
| **Sharpe / Sortino ratio** | Annualized excess return over risk-free per unit of (downside) volatility |
| **Max drawdown** | Worst peak-to-trough decline of the indexed portfolio path |
| **Concentration (HHI)** | Herfindahl-Hirschman Index over position weights + effective-N |
| **Correlation matrix** | Pairwise ρ of daily returns across all holdings |
| **Stress tests** | Beta-scaled replay of 1987, 2008, COVID-19, and 2022 market shocks |

The dashboard pairs every metric with layered explanations — plain English → formula → practitioner
caveats (fat tails, correlation breakdown in crises, backward-looking beta) — so it reads for
beginners and holds up to expert scrutiny.

## Architecture

```
┌─────────────┐   CSV upload    ┌──────────────────┐   one batched call   ┌───────────────┐
│  Streamlit   │ ──────────────► │  Analytics engine │ ───────────────────► │ Yahoo Finance │
│  frontend    │ ◄────────────── │  (pandas/NumPy)   │ ◄─────────────────── │  (yfinance)   │
└─────────────┘   full report   └──────────────────┘   1y daily closes    └───────────────┘
       │                                 ▲
       │      REST (JSON), API-key gate  │
       └────────► FastAPI service ───────┘
```

- **`backend/analytics.py`** — pure, stateless quantitative engine (single tested entry point: `portfolio_report`)
- **`backend/main.py`** — FastAPI service exposing the engine as a documented REST API (`/docs`)
- **`frontend/app.py`** — Streamlit UI: API-first, falls back to the local engine so it runs standalone
- **Fault-tolerant market data**: every Yahoo call goes through one hardened fetcher that survives
  rate limits and empty responses; the app degrades to CSV prices with a visible banner instead of crashing

## Run it locally

```bash
pip install -r requirements.txt
streamlit run frontend/app.py            # standalone risk dashboard (no API needed)
streamlit run modeling/model.py          # IB/PE modeling suite (three-statement, M&A, LBO)

# optional: run the REST API too
uvicorn backend.main:app --reload        # docs at http://localhost:8000/docs
```

Upload any CSV with columns `Ticker, Shares, Average_Price, Current_Price, Tier` — or use the built-in
demo portfolio. Live prices refresh automatically (15-minute cache).

## Tests

```bash
pytest tests/ -v
```

The suite stubs all market data, so it runs offline and in CI — covering validation, HHI math,
the live-data path, and every degradation path (Yahoo down, benchmark-only downloads, bad rows).
The modeling engines add deterministic tests that enforce **balance-sheet integrity, statement
articulation, and the M&A/LBO arithmetic** (`tests/test_modeling.py`) — 28 tests total.

## IB/PE Modeling Suite (`modeling/`)

The corporate-finance counterpart to the risk tools — the models an investment-banking or
private-equity analyst actually builds. It ships as a **second page of the Market Command Center**
(`analyzer/dashboard.py` mounts it via `st.navigation`, so it auto-deploys on push-to-main) and also
runs standalone with `streamlit run modeling/model.py`. Pure, unit-tested engines under a thin UI:

| Tab | What it does |
|---|---|
| **Three-Statement Model** | Driver-based, fully-linked IS/BS/CFS that *articulates* and **balances to the penny** — a live check proves `Assets − (L+E) ≈ 0` in every projected year. Revolver cash-plug; interest on beginning-of-period debt to avoid circular references. |
| **M&A / Accretion-Dilution** | Sources & uses, pro-forma EPS, accretion/(dilution) %, breakeven synergies, goodwill, ownership split, and premium×mix / synergies×premium sensitivity heatmaps. |
| **LBO / Returns** | Entry leverage → free-cash-flow debt sweep → exit, with IRR/MOIC, a value-creation waterfall (EBITDA growth vs. multiple vs. deleveraging), and an exit-multiple × leverage IRR grid. |

Engines live in `modeling/three_statement.py`, `ma_model.py`, and `lbo_model.py`; correctness is
enforced by `tests/test_modeling.py`.

## Investment memos (`research/`)

A short pitch book that pairs with the suite — a **long** (Visa), a **short/avoid** (Tesla), and a
**PE buyout** (US Foods), each fact-checked against primary filings and dated. See
[`research/`](research/).

## Honest limitations

Parametric VaR assumes normal returns (real tails are fatter — the app shows the historical-vs-parametric
gap explicitly). Stress tests are first-order beta-scaling, ignoring convexity and correlation breakdown.
The backtest applies today's weights to the past year without rebalancing or dividends. Educational
tool — not investment advice.

---

Built by **Jeriel De Leon** — [jerieldeleon.netlify.app](https://jerieldeleon.netlify.app)
