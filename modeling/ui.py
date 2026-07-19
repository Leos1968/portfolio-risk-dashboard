"""Rendering for the IB/PE Modeling Suite.

The entire UI lives in `render()` so it can be mounted two ways with no
duplication: as a standalone Streamlit app (`modeling/model.py`) and as a page
inside the deployed Market Command Center (`analyzer/ibpe_modeling.py`). The
caller owns `st.set_page_config`; this module never sets it.

All math lives in the pure, unit-tested engines (`three_statement.py`,
`ma_model.py`, `lbo_model.py`); this file is presentation only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modeling.three_statement import Assumptions, BaseFinancials, build_model
from modeling.ma_model import (Acquirer, DealTerms, Target, run_deal,
                               sensitivity_premium_stock, sensitivity_synergies_premium)
from modeling.lbo_model import LBOAssumptions, run_lbo, irr_sensitivity
from modeling.presets import THREE_STATEMENT_PRESETS, MA_PRESETS, LBO_PRESETS

GITHUB_URL = "https://github.com/Leos1968/portfolio-risk-dashboard"
AUTHOR_URL = "https://jerieldeleon.netlify.app"

# Validated dark-surface palette (shared with the risk dashboards)
BLUE, GOOD, RED, CRITICAL = "#3987e5", "#0ca30c", "#e66767", "#d03b3b"
AMBER, MUTED, GRID, BASELINE, INK = "#e0a458", "#898781", "#2c2c2a", "#383835", "#fafafa"
DIVERGING = [[0.0, CRITICAL], [0.5, "#1a1a19"], [1.0, GOOD]]


def _style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(t=36, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="system-ui, 'Segoe UI', sans-serif"),
        hoverlabel=dict(bgcolor="#1a1a19", font_color=INK, bordercolor=BASELINE),
        legend=dict(orientation="h", y=1.12, font=dict(color=MUTED)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BASELINE, tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, tickfont=dict(color=MUTED), zeroline=False)
    return fig


def _statement(df: pd.DataFrame, money_fmt: str = "{:,.1f}", bold: tuple[str, ...] = ()):
    """Render a statement DataFrame (line items × years) as a styled table."""
    def _emph(row):
        strong = any(k in row.name for k in bold)
        return ["font-weight: 700; color: #ffffff" if strong else "" for _ in row]

    styled = (df.style.format(money_fmt, na_rep="—")
              .apply(_emph, axis=1)
              .set_properties(**{"font-size": "13px"}))
    st.dataframe(styled, use_container_width=True)


def _heatmap(df: pd.DataFrame, xlab: str, ylab: str, height: int = 340) -> go.Figure:
    """Diverging heatmap (red dilutive / green accretive) with cell labels."""
    z = df.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z, x=list(df.columns), y=list(df.index),
        colorscale=DIVERGING, zmid=0.0,
        text=[[f"{v:+.1f}" for v in row] for row in z],
        texttemplate="%{text}", textfont={"size": 11, "color": INK},
        hovertemplate=f"{xlab}: %{{x}}<br>{ylab}: %{{y}}<br>%{{z:+.2f}}%<extra></extra>",
        colorbar=dict(title="%", outlinewidth=0, tickfont=dict(color=MUTED)),
    ))
    fig.update_layout(xaxis_title=xlab, yaxis_title=ylab)
    fig.update_yaxes(autorange="reversed")
    return _style(fig, height)


def _fmt(x: float, prefix: str = "$", suffix: str = "", dp: int = 0) -> str:
    return f"{prefix}{x:,.{dp}f}{suffix}"


def render() -> None:
    """Draw the whole suite. Caller must have already set page config."""
    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    st.sidebar.title("📐 IB/PE Modeling Suite")
    st.sidebar.caption(
        "The three models an analyst actually builds — three-statement, "
        "accretion/dilution, and LBO — with the accounting done right."
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        f"**Explore more**\n\n"
        f"[🏛️ Institutional Risk Dashboard](https://institutional-risk-dashboard.streamlit.app)  \n"
        f"[⭐ Source code on GitHub]({GITHUB_URL})  \n"
        f"[👤 Built by Jeriel De Leon]({AUTHOR_URL})"
    )
    st.sidebar.caption("Educational tool — not investment advice. Figures in the currency you enter (e.g. $ millions).")

    st.title("📐 IB/PE Modeling Suite")

    tab_ts, tab_ma, tab_lbo, tab_learn = st.tabs(
        ["📐 Three-Statement Model", "🤝 M&A / Accretion-Dilution", "🏦 LBO / Returns", "🧭 Methodology"]
    )

    # =======================================================================
    # 📐 Three-Statement Model
    # =======================================================================
    with tab_ts:
        st.subheader("Driver-based, fully-linked three-statement model")
        st.caption("Set the operating drivers; the income statement, balance sheet, and cash-flow "
                   "statement are built to *articulate* — and the balance sheet balances to the penny.")

        preset_name = st.selectbox("Start from a template", list(THREE_STATEMENT_PRESETS.keys()), key="ts_preset")
        p = THREE_STATEMENT_PRESETS[preset_name]
        b0: BaseFinancials = p["base"]
        a0: Assumptions = p["assumptions"]
        k = preset_name  # widget-key salt so switching templates reloads defaults

        st.markdown("##### Operating assumptions")
        horizon = st.slider("Projection horizon (years)", 3, 7, a0.years, key=f"ts_years_{k}")
        c1, c2, c3, c4 = st.columns(4)
        growth = c1.number_input("Revenue growth %", -20.0, 60.0, a0.revenue_growth * 100, 0.5, key=f"ts_g_{k}") / 100
        gm = c2.number_input("Gross margin %", 5.0, 95.0, a0.gross_margin * 100, 0.5, key=f"ts_gm_{k}") / 100
        sga = c3.number_input("SG&A % of revenue", 0.0, 70.0, a0.sga_pct * 100, 0.5, key=f"ts_sga_{k}") / 100
        rd = c4.number_input("R&D % of revenue", 0.0, 50.0, a0.rd_pct * 100, 0.5, key=f"ts_rd_{k}") / 100
        c1, c2, c3, c4 = st.columns(4)
        da = c1.number_input("D&A % of revenue", 0.0, 30.0, a0.da_pct * 100, 0.5, key=f"ts_da_{k}") / 100
        capex = c2.number_input("Capex % of revenue", 0.0, 40.0, a0.capex_pct * 100, 0.5, key=f"ts_capex_{k}") / 100
        tax = c3.number_input("Tax rate %", 0.0, 50.0, a0.tax_rate * 100, 0.5, key=f"ts_tax_{k}") / 100
        payout = c4.number_input("Dividend payout %", 0.0, 100.0, a0.payout_ratio * 100, 1.0, key=f"ts_pay_{k}") / 100

        st.markdown("##### Working capital & financing")
        c1, c2, c3, c4 = st.columns(4)
        dso = c1.number_input("DSO (receivable days)", 0.0, 200.0, float(a0.dso), 1.0, key=f"ts_dso_{k}")
        dio = c2.number_input("DIO (inventory days)", 0.0, 300.0, float(a0.dio), 1.0, key=f"ts_dio_{k}")
        dpo = c3.number_input("DPO (payable days)", 0.0, 200.0, float(a0.dpo), 1.0, key=f"ts_dpo_{k}")
        ir = c4.number_input("Interest rate on debt %", 0.0, 20.0, a0.interest_rate * 100, 0.25, key=f"ts_ir_{k}") / 100
        c1, c2, c3, c4 = st.columns(4)
        amort = c1.number_input("Mandatory debt repayment / yr", 0.0, 1e6, float(a0.mandatory_amort), 10.0, key=f"ts_amort_{k}")
        sweep = c2.number_input("Cash sweep % of surplus", 0.0, 100.0, a0.cash_sweep_pct * 100, 5.0, key=f"ts_sweep_{k}") / 100
        min_cash = c3.number_input("Minimum cash (revolver floor)", 0.0, 1e6, float(a0.min_cash), 10.0, key=f"ts_mincash_{k}")
        use_rev = c4.checkbox("Use revolver as cash plug", value=a0.use_revolver, key=f"ts_rev_{k}")

        with st.expander("⚙️ Base-year (Year 0) starting point — edit to model a real company"):
            st.caption("Enter the latest reported year from a 10-K. Any imbalance is absorbed into a "
                       "labelled *Other assets (plug)* so the model always opens balanced.")
            c1, c2, c3 = st.columns(3)
            rev0 = c1.number_input("Revenue", 0.0, 1e7, float(b0.revenue), 10.0, key=f"b_rev_{k}")
            cogs0 = c2.number_input("COGS", 0.0, 1e7, float(b0.cogs), 10.0, key=f"b_cogs_{k}")
            sga0 = c3.number_input("SG&A", 0.0, 1e7, float(b0.sga), 10.0, key=f"b_sga_{k}")
            c1, c2, c3 = st.columns(3)
            rd0 = c1.number_input("R&D", 0.0, 1e7, float(b0.rd), 10.0, key=f"b_rd_{k}")
            da0 = c2.number_input("D&A", 0.0, 1e7, float(b0.da), 10.0, key=f"b_da_{k}")
            cash0 = c3.number_input("Cash", 0.0, 1e7, float(b0.cash), 10.0, key=f"b_cash_{k}")
            c1, c2, c3 = st.columns(3)
            ar0 = c1.number_input("Accounts receivable", 0.0, 1e7, float(b0.accounts_receivable), 10.0, key=f"b_ar_{k}")
            inv0 = c2.number_input("Inventory", 0.0, 1e7, float(b0.inventory), 10.0, key=f"b_inv_{k}")
            ppe0 = c3.number_input("Net PP&E", 0.0, 1e7, float(b0.net_ppe), 10.0, key=f"b_ppe_{k}")
            c1, c2, c3 = st.columns(3)
            oa0 = c1.number_input("Other assets", 0.0, 1e7, float(b0.other_assets), 10.0, key=f"b_oa_{k}")
            ap0 = c2.number_input("Accounts payable", 0.0, 1e7, float(b0.accounts_payable), 10.0, key=f"b_ap_{k}")
            debt0 = c3.number_input("Term debt", 0.0, 1e7, float(b0.debt), 10.0, key=f"b_debt_{k}")
            c1, c2, c3 = st.columns(3)
            ol0 = c1.number_input("Other liabilities", 0.0, 1e7, float(b0.other_liabilities), 10.0, key=f"b_ol_{k}")
            cs0 = c2.number_input("Common stock / APIC", 0.0, 1e7, float(b0.common_stock), 10.0, key=f"b_cs_{k}")
            re0 = c3.number_input("Retained earnings", -1e7, 1e7, float(b0.retained_earnings), 10.0, key=f"b_re_{k}")

        base = BaseFinancials(
            revenue=rev0, cogs=cogs0, sga=sga0, rd=rd0, da=da0, cash=cash0,
            accounts_receivable=ar0, inventory=inv0, net_ppe=ppe0, other_assets=oa0,
            accounts_payable=ap0, debt=debt0, revolver=0.0, other_liabilities=ol0,
            common_stock=cs0, retained_earnings=re0,
        )
        assumptions = Assumptions(
            years=horizon, revenue_growth=growth, gross_margin=gm, sga_pct=sga, rd_pct=rd,
            da_pct=da, capex_pct=capex, tax_rate=tax, dso=dso, dio=dio, dpo=dpo,
            interest_rate=ir, revolver_rate=a0.revolver_rate, cash_yield=a0.cash_yield,
            min_cash=min_cash, mandatory_amort=amort, cash_sweep_pct=sweep,
            payout_ratio=payout, use_revolver=use_rev,
        )
        model = build_model(base, assumptions).rounded(1)

        if model.balances:
            st.success(f"✅ Balance sheet balances in every year — max |Assets − (L+E)| = "
                       f"{model.max_balance_error:.2e}. The three statements articulate.")
        else:
            st.error(f"⚠️ Balance check off by {model.max_balance_error:,.2f} — check the base-year inputs.")

        m = model.metrics
        last = model.columns[-1]
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric(f"Revenue ({last})", _fmt(model.income_statement.loc["Revenue", last]))
        r2.metric(f"EBITDA ({last})", _fmt(model.income_statement.loc["— Memo: EBITDA", last]))
        r3.metric(f"EBITDA margin ({last})", f"{m.loc['EBITDA Margin %', last]:.1f}%")
        r4.metric(f"Net income ({last})", _fmt(model.income_statement.loc["Net Income", last]))
        r5.metric(f"Net debt / EBITDA ({last})", f"{m.loc['Net Debt / EBITDA', last]:.2f}x")

        cc1, cc2 = st.columns(2)
        with cc1:
            yrs = model.columns
            fig = go.Figure()
            fig.add_bar(x=yrs, y=model.income_statement.loc["Revenue"], name="Revenue", marker_color=BLUE)
            fig.add_bar(x=yrs, y=model.income_statement.loc["— Memo: EBITDA"], name="EBITDA", marker_color=GOOD)
            fig.update_layout(barmode="group", title="Revenue & EBITDA", yaxis_title="")
            st.plotly_chart(_style(fig, 340), use_container_width=True)
        with cc2:
            fig = go.Figure()
            for row, color in [("Gross Margin %", MUTED), ("EBITDA Margin %", GOOD), ("Net Margin %", BLUE)]:
                fig.add_trace(go.Scatter(x=yrs, y=m.loc[row], name=row.replace(" %", ""),
                                         mode="lines+markers", line={"color": color, "width": 2.5}))
            fig.update_layout(title="Margin trajectory", yaxis_title="%")
            st.plotly_chart(_style(fig, 340), use_container_width=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            fig = go.Figure()
            fig.add_bar(x=yrs, y=model.balance_sheet.loc["Cash & Equivalents"], name="Cash", marker_color=BLUE)
            fig.add_trace(go.Scatter(x=yrs, y=m.loc["Free Cash Flow"], name="Free Cash Flow",
                                     mode="lines+markers", line={"color": GOOD, "width": 2.5}))
            fig.update_layout(title="Cash & free cash flow", yaxis_title="")
            st.plotly_chart(_style(fig, 320), use_container_width=True)
        with cc2:
            fig = go.Figure()
            fig.add_bar(x=yrs, y=m.loc["Net Debt"], name="Net Debt", marker_color=MUTED)
            fig.add_trace(go.Scatter(x=yrs, y=m.loc["Net Debt / EBITDA"], name="Net Debt / EBITDA",
                                     mode="lines+markers", yaxis="y2", line={"color": AMBER, "width": 2.5}))
            fig.update_layout(title="Leverage", yaxis_title="Net debt",
                              yaxis2=dict(title="x EBITDA", overlaying="y", side="right",
                                          showgrid=False, tickfont=dict(color=MUTED)))
            st.plotly_chart(_style(fig, 320), use_container_width=True)

        st.markdown("##### The three statements")
        t1, t2, t3, t4 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow", "Key Metrics"])
        with t1:
            _statement(model.income_statement, bold=("Revenue", "Gross Profit", "EBIT", "Net Income"))
        with t2:
            _statement(model.balance_sheet, bold=("Total Assets", "Total Liabilities & Equity", "Check"))
        with t3:
            _statement(model.cash_flow, bold=("Cash Flow from", "Net Change", "Ending Cash"))
        with t4:
            _statement(model.metrics, money_fmt="{:,.2f}")

        with st.expander("📖 How the three statements link (and why this one balances)"):
            st.markdown(
                "A model is *three-statement* only if the statements feed each other:\n\n"
                "- **Net income** (income statement) flows to **retained earnings** on the balance sheet "
                "*and* to the top of the **cash-flow statement**.\n"
                "- **D&A** is a non-cash expense: subtracted on the P&L, added back on the CFS, and it "
                "depreciates **net PP&E** on the balance sheet (PP&E rolls: *begin + capex − D&A*).\n"
                "- **Working capital** — receivables, inventory, payables — is driven off DSO/DIO/DPO and "
                "its *change* is a cash flow.\n"
                "- The **ending cash** on the cash-flow statement **is** the cash line on the balance sheet.\n\n"
                "Build the cash-flow statement from those pieces and the balance sheet balances *by "
                "construction* — Assets ≡ Liabilities + Equity — which is why the check row above is zero "
                "in every year. A revolver is the plug that keeps cash at your minimum; surplus cash sweeps "
                "debt down.\n\n"
                "**Conventions (stated honestly):** interest is charged on *beginning-of-period* debt so the "
                "model has no circular reference; taxes are floored at zero (no NOL carryforwards); and "
                "'Other assets/liabilities' are held flat. These are the standard simplifications for a clean "
                "transparent model — a live deal model would layer in a full debt schedule, deferred taxes, "
                "and stock-based comp."
            )

    # =======================================================================
    # 🤝 M&A / Accretion-Dilution
    # =======================================================================
    with tab_ma:
        st.subheader("Accretion / dilution — does the deal add to EPS?")
        st.caption("The banker's first-pass test on any acquisition. Set the two companies and the deal "
                   "structure; the model builds pro-forma EPS and tells you whether it is accretive.")

        ma_preset = st.selectbox("Start from a template", list(MA_PRESETS.keys()), key="ma_preset")
        mp = MA_PRESETS[ma_preset]
        acq0: Acquirer = mp["acquirer"]
        tgt0: Target = mp["target"]
        dt0: DealTerms = mp["terms"]
        k = ma_preset

        ac, tc = st.columns(2)
        with ac:
            st.markdown("##### Acquirer")
            acq_ni = st.number_input("Net income (NTM)", 0.0, 1e7, float(acq0.net_income), 10.0, key=f"a_ni_{k}")
            acq_sh = st.number_input("Diluted shares", 0.1, 1e7, float(acq0.shares), 1.0, key=f"a_sh_{k}")
            acq_px = st.number_input("Share price", 0.01, 1e6, float(acq0.share_price), 0.5, key=f"a_px_{k}")
            acq_tax = st.number_input("Tax rate %", 0.0, 50.0, acq0.tax_rate * 100, 0.5, key=f"a_tax_{k}") / 100
            acq_nd = st.number_input("Net debt", -1e7, 1e7, float(acq0.net_debt), 10.0, key=f"a_nd_{k}")
            acq_eb = st.number_input("EBITDA (for leverage)", 0.0, 1e7, float(acq0.ebitda), 10.0, key=f"a_eb_{k}")
        with tc:
            st.markdown("##### Target")
            tgt_ni = st.number_input("Net income (NTM)", 0.0, 1e7, float(tgt0.net_income), 5.0, key=f"t_ni_{k}")
            tgt_sh = st.number_input("Diluted shares", 0.1, 1e7, float(tgt0.shares), 1.0, key=f"t_sh_{k}")
            tgt_px = st.number_input("Unaffected share price", 0.01, 1e6, float(tgt0.share_price), 0.5, key=f"t_px_{k}")
            tgt_nd = st.number_input("Net debt", -1e7, 1e7, float(tgt0.net_debt), 5.0, key=f"t_nd_{k}")
            tgt_eb = st.number_input("EBITDA (for leverage)", 0.0, 1e7, float(tgt0.ebitda), 5.0, key=f"t_eb_{k}")
            tgt_be = st.number_input("Book equity (for goodwill)", -1e7, 1e7, float(tgt0.book_equity), 5.0, key=f"t_be_{k}")

        st.markdown("##### Deal structure")
        c1, c2, c3 = st.columns(3)
        prem = c1.number_input("Offer premium %", 0.0, 150.0, dt0.offer_premium * 100, 1.0, key=f"d_prem_{k}") / 100
        ndr = c2.number_input("New debt rate %", 0.0, 20.0, dt0.new_debt_rate * 100, 0.25, key=f"d_ndr_{k}") / 100
        cy = c3.number_input("Foregone cash yield %", 0.0, 20.0, dt0.cash_yield * 100, 0.25, key=f"d_cy_{k}") / 100
        c1, c2, c3 = st.columns(3)
        p_stock = c1.slider("% Stock", 0, 100, int(dt0.pct_stock * 100), key=f"d_stk_{k}")
        p_cash = c2.slider("% Cash", 0, 100, int(dt0.pct_cash * 100), key=f"d_csh_{k}")
        p_debt = c3.slider("% Debt", 0, 100, int(dt0.pct_debt * 100), key=f"d_dbt_{k}")
        mix = p_stock + p_cash + p_debt
        if mix == 0:
            st.warning("Set a consideration mix (stock + cash + debt).")
            st.stop()
        p_stock, p_cash, p_debt = p_stock / mix, p_cash / mix, p_debt / mix  # normalize to 100%
        c1, c2 = st.columns(2)
        syn = c1.number_input("Pre-tax annual synergies", 0.0, 1e7, float(dt0.pretax_synergies), 5.0, key=f"d_syn_{k}")
        fees = c2.number_input("Transaction fees", 0.0, 1e6, float(dt0.transaction_fees), 5.0, key=f"d_fee_{k}")

        acq = Acquirer(acq_ni, acq_sh, acq_px, acq_tax, acq_nd, acq_eb)
        tgt = Target(tgt_ni, tgt_sh, tgt_px, tgt_nd, tgt_eb, tgt_be)
        terms = DealTerms(prem, p_stock, p_cash, p_debt, ndr, cy, syn, True, fees)
        r = run_deal(acq, tgt, terms)

        st.caption(f"Mix normalized to **{p_stock:.0%} stock / {p_cash:.0%} cash / {p_debt:.0%} debt**. "
                   f"Offer **{_fmt(r.offer_price, dp=2)}**/sh ({prem:.0%} premium) · "
                   f"equity value **{_fmt(r.equity_purchase_price)}** · EV **{_fmt(r.enterprise_value)}**.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Acquirer standalone EPS", _fmt(r.standalone_eps, dp=2))
        m2.metric("Pro-forma EPS", _fmt(r.proforma_eps, dp=2), f"{r.accretion_dilution:+.1%}")
        tag = "ACCRETIVE" if r.accretion_dilution > 1e-9 else ("DILUTIVE" if r.accretion_dilution < -1e-9 else "NEUTRAL")
        m3.metric("Verdict", tag, f"{r.accretion_dilution:+.2%} to EPS")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("New shares issued", _fmt(r.new_shares, prefix="", dp=1))
        m2.metric("Goodwill created", _fmt(r.goodwill))
        m3.metric("Target holders own", f"{r.target_ownership:.1%}")
        m4.metric("Pro-forma leverage", f"{r.proforma_leverage:.2f}x" if np.isfinite(r.proforma_leverage) else "—")

        if r.breakeven_pretax_synergies > 0:
            st.info(f"💡 This deal needs **{_fmt(r.breakeven_pretax_synergies)}** of pre-tax annual synergies "
                    f"to break even on EPS. You have assumed {_fmt(syn)}.")
        else:
            st.info("💡 The deal is accretive with **zero** synergies — the currency/financing is cheap "
                    "relative to the target's earnings yield.")

        su_col, sens_col = st.columns([1, 1.4])
        with su_col:
            st.markdown("##### Sources & Uses")
            su = r.sources_uses
            rows = [{"Sources": f"{s[0]}", "$": s[1], "Uses": u[0], "$ ": u[1]}
                    for s, u in zip(su["Sources"], su["Uses"])]
            st.dataframe(pd.DataFrame(rows).style.format({"$": "{:,.0f}", "$ ": "{:,.0f}"}),
                         use_container_width=True, hide_index=True)
        with sens_col:
            st.markdown("##### Accretion/(dilution) — premium × % stock")
            premiums = [round(prem + d, 2) for d in (-0.15, -0.075, 0.0, 0.075, 0.15, 0.225)]
            premiums = [max(0.0, x) for x in premiums]
            stock_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
            grid = sensitivity_premium_stock(acq, tgt, terms, premiums, stock_grid)
            st.plotly_chart(_heatmap(grid, "% Stock", "Premium"), use_container_width=True)

        st.markdown("##### Accretion/(dilution) — synergies × premium")
        syn_grid = sorted({0.0, syn, syn * 2 if syn else 50.0, syn * 3 if syn else 100.0,
                           r.breakeven_pretax_synergies if r.breakeven_pretax_synergies > 0 else 150.0})
        syn_grid = [round(s, 0) for s in syn_grid if s >= 0][:6]
        prem_grid = [round(max(0.0, prem + d), 2) for d in (-0.1, 0.0, 0.1, 0.2, 0.3)]
        st.plotly_chart(_heatmap(sensitivity_synergies_premium(acq, tgt, terms, syn_grid, prem_grid),
                                 "Premium", "Pre-tax synergies"), use_container_width=True)

        with st.expander("📖 The intuition — the P/E rule of thumb"):
            st.markdown(
                "Pro-forma EPS = *(combined net income + after-tax synergies − after-tax financing cost) "
                "÷ (acquirer shares + new shares issued)*.\n\n"
                "The shortcut every banker knows: in an **all-stock** deal you are swapping your currency "
                "for the target's earnings, so the deal is **accretive whenever your P/E is higher than the "
                "P/E you pay** for the target. In a **cash or debt** deal, it is accretive when the target's "
                "earnings yield (E/P on the purchase price) exceeds the **after-tax** cost of the cash or "
                "debt you use. High-multiple acquirers with cheap financing accrete; low-multiple acquirers "
                "reaching for a pricey target dilute.\n\n"
                "**What this first-pass ignores:** purchase-accounting step-ups and D&A, one-time integration "
                "and restructuring costs, the phasing of synergies, and any revenue dis-synergies. EPS "
                "accretion is also *not* the same as value creation — a deal can be accretive and still "
                "destroy value if you overpay."
            )

    # =======================================================================
    # 🏦 LBO / Returns
    # =======================================================================
    with tab_lbo:
        st.subheader("Leveraged buyout — IRR & MOIC")
        st.caption("The private-equity lens: buy with debt, pay it down with the company's cash flow, grow "
                   "EBITDA, and sell in ~5 years. See which lever the returns actually come from.")

        lbo_name = st.selectbox("Start from a template", list(LBO_PRESETS.keys()), key="lbo_preset")
        l0: LBOAssumptions = LBO_PRESETS[lbo_name]
        k = lbo_name

        c1, c2, c3, c4 = st.columns(4)
        rev = c1.number_input("Entry revenue", 1.0, 1e7, float(l0.entry_revenue), 10.0, key=f"l_rev_{k}")
        margin = c2.number_input("EBITDA margin %", 1.0, 80.0, (l0.ebitda_margin if isinstance(l0.ebitda_margin, float) else l0.ebitda_margin[0]) * 100, 0.5, key=f"l_mar_{k}") / 100
        ent_mult = c3.number_input("Entry EV/EBITDA", 2.0, 25.0, float(l0.entry_multiple), 0.5, key=f"l_em_{k}")
        ext_mult = c4.number_input("Exit EV/EBITDA", 2.0, 25.0, float(l0.exit_multiple), 0.5, key=f"l_xm_{k}")
        c1, c2, c3, c4 = st.columns(4)
        lev = c1.number_input("Entry leverage (x EBITDA)", 0.0, 8.0, float(l0.entry_leverage), 0.25, key=f"l_lev_{k}")
        grow = c2.number_input("Revenue growth %", -10.0, 40.0, (l0.revenue_growth if isinstance(l0.revenue_growth, float) else l0.revenue_growth[0]) * 100, 0.5, key=f"l_g_{k}") / 100
        hold = c3.number_input("Hold period (years)", 3, 7, int(l0.years), 1, key=f"l_yr_{k}")
        debt_rate = c4.number_input("Debt interest rate %", 0.0, 20.0, l0.debt_rate * 100, 0.25, key=f"l_dr_{k}") / 100
        c1, c2, c3, c4 = st.columns(4)
        capex_p = c1.number_input("Capex % of revenue", 0.0, 30.0, (l0.capex_pct if isinstance(l0.capex_pct, float) else l0.capex_pct[0]) * 100, 0.5, key=f"l_cx_{k}") / 100
        da_p = c2.number_input("D&A % of revenue", 0.0, 30.0, (l0.da_pct if isinstance(l0.da_pct, float) else l0.da_pct[0]) * 100, 0.5, key=f"l_da_{k}") / 100
        sweep = c3.number_input("Cash sweep %", 0.0, 100.0, l0.cash_sweep_pct * 100, 5.0, key=f"l_sw_{k}") / 100
        fees_p = c4.number_input("Fees % of EV", 0.0, 10.0, l0.fees_pct * 100, 0.25, key=f"l_fe_{k}") / 100

        la = LBOAssumptions(
            entry_revenue=rev, entry_multiple=ent_mult, exit_multiple=ext_mult, entry_leverage=lev,
            years=int(hold), ebitda_margin=margin, revenue_growth=grow, da_pct=da_p, capex_pct=capex_p,
            nwc_pct=l0.nwc_pct, tax_rate=l0.tax_rate, debt_rate=debt_rate,
            mandatory_amort_pct=l0.mandatory_amort_pct, cash_sweep_pct=sweep, fees_pct=fees_p,
            min_cash=l0.min_cash,
        )
        lr = run_lbo(la)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("IRR", f"{lr.irr:.1%}" if np.isfinite(lr.irr) else "—")
        m2.metric("MOIC", f"{lr.moic:.2f}x" if np.isfinite(lr.moic) else "—")
        m3.metric("Sponsor equity", _fmt(lr.sponsor_equity))
        m4.metric("Entry → exit leverage", f"{lr.entry_leverage:.1f}x → {lr.exit_leverage:.1f}x")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Enterprise value", _fmt(lr.enterprise_value))
        m2.metric("New debt", _fmt(lr.new_debt))
        m3.metric("Entry EBITDA", _fmt(lr.entry_ebitda))
        m4.metric("Exit equity value", _fmt(lr.exit_equity))

        cc1, cc2 = st.columns([1.1, 1])
        with cc1:
            st.markdown("##### Value-creation bridge")
            keys = ["EBITDA growth", "Multiple change", "Debt paydown", "Fees"]
            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute"] + ["relative"] * 4 + ["total"],
                x=["Sponsor equity"] + keys + ["Exit equity"],
                y=[lr.sponsor_equity] + [lr.bridge[key] for key in keys] + [lr.exit_equity],
                text=[_fmt(lr.sponsor_equity)] + [f"{lr.bridge[key]:+,.0f}" for key in keys] + [_fmt(lr.exit_equity)],
                textposition="outside",
                connector={"line": {"color": BASELINE}},
                increasing={"marker": {"color": GOOD}},
                decreasing={"marker": {"color": RED}},
                totals={"marker": {"color": BLUE}},
            ))
            fig.update_layout(title="How sponsor equity grows to exit", yaxis_title="")
            st.plotly_chart(_style(fig, 360), use_container_width=True)
        with cc2:
            st.markdown("##### IRR sensitivity — exit multiple × entry leverage")
            xmults = [round(ent_mult + d, 1) for d in (-2, -1, 0, 1, 2)]
            levs = [round(x, 1) for x in (max(0.0, lev - 1.5), max(0.0, lev - 0.75), lev, lev + 0.75, lev + 1.5)]
            st.plotly_chart(_heatmap(irr_sensitivity(la, xmults, levs), "Entry leverage", "Exit multiple", 360),
                            use_container_width=True)

        st.markdown("##### Debt paydown & cash flow schedule")
        _statement(lr.schedule, money_fmt="{:,.1f}",
                   bold=("EBITDA", "Free Cash Flow", "Net Debt (end)"))

        with st.expander("📖 Where LBO returns come from"):
            st.markdown(
                "A buyout makes money three ways, and a sharp interviewer will ask which one you are "
                "relying on:\n\n"
                "1. **Operations** — growing EBITDA (revenue × margin) at the entry multiple.\n"
                "2. **Multiple expansion** — selling for a higher EV/EBITDA than you paid. The riskiest lever, "
                "because it depends on the market, not on you — good memos assume flat-to-conservative "
                "multiples.\n"
                "3. **Deleveraging** — the company's free cash flow sweeps the debt down, so a growing slice "
                "of a fixed enterprise value accrues to your equity.\n\n"
                "**MOIC** is exit equity ÷ the equity you put in; **IRR** annualizes it over the hold. Leverage "
                "amplifies both — and amplifies the downside, which is why the debt schedule and interest "
                "coverage matter as much as the return.\n\n"
                "**Simplifications here:** a single blended debt tranche (a real model splits revolver / term "
                "loan / bonds with their own rates and covenants), interest on beginning debt, no dividend "
                "recapitalizations, and no management-equity rollover or option pool."
            )

    # =======================================================================
    # 🧭 Methodology
    # =======================================================================
    with tab_learn:
        st.subheader("Why these three models")
        st.markdown(
            "Risk analytics tell you how a *portfolio* behaves. These tools do the other half of the job — "
            "they value and structure *a single company or deal*, which is the daily work of investment "
            "banking and private equity. The three build on each other: a three-statement model produces the "
            "cash flows that a DCF or an LBO discounts, and an accretion/dilution model is a one-year "
            "three-statement combination of two companies. "
            f"For the market-risk side, see the [sister apps]({GITHUB_URL})."
        )

        st.subheader("Glossary")
        glossary = [
            ("Three-statement model", "A linked income statement, balance sheet, and cash-flow statement where "
             "net income, D&A, and working capital tie the three together and the balance sheet balances.",
             "https://www.wallstreetprep.com/knowledge/3-statement-model/"),
            ("Accretion / Dilution", "Whether a deal raises (accretive) or lowers (dilutive) the acquirer's "
             "earnings per share versus staying standalone.",
             "https://www.investopedia.com/terms/a/accretion.asp"),
            ("LBO", "A leveraged buyout: acquiring a company using significant borrowed money, repaid by the "
             "target's own cash flows.",
             "https://www.investopedia.com/terms/l/leveragedbuyout.asp"),
            ("IRR / MOIC", "The two headline PE return metrics: the annualized return (IRR) and the multiple of "
             "invested capital (MOIC = exit equity ÷ equity invested).",
             "https://www.investopedia.com/terms/i/irr.asp"),
            ("EV / EBITDA", "Enterprise value over EBITDA — the multiple used to price whole companies because "
             "it is capital-structure-neutral.",
             "https://www.investopedia.com/terms/e/ev-ebitda.asp"),
            ("Working capital (DSO/DIO/DPO)", "Days of sales in receivables, days of inventory, and days of "
             "payables — they set how much cash growth ties up.",
             "https://www.investopedia.com/terms/w/workingcapital.asp"),
            ("Goodwill", "The premium paid over the fair value of a target's net identifiable assets; it sits on "
             "the acquirer's balance sheet after the deal.",
             "https://www.investopedia.com/terms/g/goodwill.asp"),
        ]
        for name, plain, link in glossary:
            with st.expander(f"**{name}**"):
                st.markdown(f"{plain}\n\n[Deep dive ↗]({link})")

        st.subheader("Curated resources")
        st.markdown(
            "- [Damodaran Online](https://pages.stern.nyu.edu/~adamodar/) — NYU Stern's valuation bible, free datasets & spreadsheets\n"
            "- [SEC EDGAR](https://www.sec.gov/edgar/search/) — every 10-K/10-Q, the source for real base-year numbers\n"
            "- [Wall Street Prep](https://www.wallstreetprep.com/knowledge/) & [Macabacus](https://macabacus.com/) — modeling conventions used on the Street\n"
            "- [Mergers & Inquisitions / BIWS](https://mergersandinquisitions.com/) — how recruiting actually tests this\n"
            "- [Aswath Damodaran on YouTube](https://www.youtube.com/user/AswathDamodaran) — the definitive valuation course, free"
        )

        st.caption(
            f"Built by [Jeriel De Leon]({AUTHOR_URL}) · Python, pandas, NumPy, Streamlit, Plotly · "
            "Engines are unit-tested for balance-sheet integrity and deal arithmetic. "
            "Educational tool — not investment advice."
        )
