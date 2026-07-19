"""Illustrative preset inputs for the modeling app.

These are archetypes with realistic *shapes* (margins, leverage, working-capital
days) — not the reported financials of any specific company. They exist so a
first-time user sees a sensible, balanced model immediately and can then swap in
real numbers from a 10-K. Everything downstream is fully editable.
"""

from __future__ import annotations

try:  # works both as a package (tests) and with modeling/ on sys.path (Streamlit)
    from .lbo_model import LBOAssumptions
    from .ma_model import Acquirer, DealTerms, Target
    from .three_statement import Assumptions, BaseFinancials
except ImportError:  # pragma: no cover
    from lbo_model import LBOAssumptions
    from ma_model import Acquirer, DealTerms, Target
    from three_statement import Assumptions, BaseFinancials

# ---------------------------------------------------------------------------
# Three-statement archetypes: (BaseFinancials, Assumptions)
# ---------------------------------------------------------------------------

THREE_STATEMENT_PRESETS: dict[str, dict] = {
    "Software / SaaS (illustrative)": {
        "base": BaseFinancials(
            revenue=1200, cogs=264, sga=336, rd=240, da=48,
            cash=300, accounts_receivable=180, inventory=0, net_ppe=150,
            other_assets=260, accounts_payable=60, debt=100, revolver=0,
            other_liabilities=180, common_stock=560, retained_earnings=0,
        ),
        "assumptions": Assumptions(
            years=5, revenue_growth=0.16, gross_margin=0.78, sga_pct=0.27, rd_pct=0.18,
            da_pct=0.04, capex_pct=0.05, tax_rate=0.21, dso=55, dio=0, dpo=45,
            interest_rate=0.06, revolver_rate=0.08, cash_yield=0.03, min_cash=100,
            mandatory_amort=0, cash_sweep_pct=0.0, payout_ratio=0.0, use_revolver=True,
        ),
    },
    "Consumer staples (illustrative)": {
        "base": BaseFinancials(
            revenue=8000, cogs=5200, sga=1600, rd=0, da=300,
            cash=200, accounts_receivable=500, inventory=900, net_ppe=3000,
            other_assets=1900, accounts_payable=800, debt=2500, revolver=0,
            other_liabilities=700, common_stock=1000, retained_earnings=1500,
        ),
        "assumptions": Assumptions(
            years=5, revenue_growth=0.04, gross_margin=0.35, sga_pct=0.20, rd_pct=0.0,
            da_pct=0.038, capex_pct=0.04, tax_rate=0.24, dso=22, dio=63, dpo=56,
            interest_rate=0.05, revolver_rate=0.07, cash_yield=0.03, min_cash=150,
            mandatory_amort=100, cash_sweep_pct=0.0, payout_ratio=0.50, use_revolver=True,
        ),
    },
    "Industrial / cyclical (illustrative)": {
        "base": BaseFinancials(
            revenue=5000, cogs=3500, sga=750, rd=150, da=250,
            cash=250, accounts_receivable=760, inventory=1100, net_ppe=2600,
            other_assets=1200, accounts_payable=620, debt=1800, revolver=0,
            other_liabilities=640, common_stock=1400, retained_earnings=0,
        ),
        "assumptions": Assumptions(
            years=5, revenue_growth=0.06, gross_margin=0.30, sga_pct=0.15, rd_pct=0.03,
            da_pct=0.05, capex_pct=0.06, tax_rate=0.24, dso=55, dio=80, dpo=45,
            interest_rate=0.06, revolver_rate=0.08, cash_yield=0.03, min_cash=200,
            mandatory_amort=120, cash_sweep_pct=0.25, payout_ratio=0.25, use_revolver=True,
        ),
    },
}


# ---------------------------------------------------------------------------
# M&A presets: (Acquirer, Target, DealTerms)
# ---------------------------------------------------------------------------

MA_PRESETS: dict[str, dict] = {
    "Large-cap stock-heavy merger (illustrative)": {
        "acquirer": Acquirer(net_income=5000, shares=2000, share_price=50, tax_rate=0.23,
                             net_debt=3000, ebitda=9000),   # EPS $2.50, ~20x P/E
        "target": Target(net_income=800, shares=500, share_price=24, net_debt=1200,
                         ebitda=1600, book_equity=3000),    # EPS $1.60, ~15x P/E
        "terms": DealTerms(offer_premium=0.30, pct_stock=0.70, pct_cash=0.15, pct_debt=0.15,
                           new_debt_rate=0.06, cash_yield=0.03, pretax_synergies=200,
                           assume_target_debt=True, transaction_fees=150),
    },
    "Cash-and-debt bolt-on (illustrative)": {
        "acquirer": Acquirer(net_income=3200, shares=800, share_price=90, tax_rate=0.23,
                             net_debt=1500, ebitda=5200),   # EPS $4.00, ~22.5x P/E
        "target": Target(net_income=260, shares=120, share_price=30, net_debt=200,
                         ebitda=520, book_equity=700),      # EPS $2.17, ~14x P/E
        "terms": DealTerms(offer_premium=0.35, pct_stock=0.0, pct_cash=0.40, pct_debt=0.60,
                           new_debt_rate=0.07, cash_yield=0.035, pretax_synergies=60,
                           assume_target_debt=True, transaction_fees=40),
    },
}


# ---------------------------------------------------------------------------
# LBO presets
# ---------------------------------------------------------------------------

LBO_PRESETS: dict[str, LBOAssumptions] = {
    "Business-services buyout (illustrative)": LBOAssumptions(
        entry_revenue=2000, entry_multiple=11.0, exit_multiple=11.0, entry_leverage=5.5,
        years=5, ebitda_margin=0.22, revenue_growth=0.06, da_pct=0.04, capex_pct=0.035,
        nwc_pct=0.08, tax_rate=0.23, debt_rate=0.09, mandatory_amort_pct=0.05,
        cash_sweep_pct=0.75, fees_pct=0.025, min_cash=0.0,
    ),
    "Consumer / retail buyout (illustrative)": LBOAssumptions(
        entry_revenue=4500, entry_multiple=9.0, exit_multiple=9.0, entry_leverage=4.5,
        years=5, ebitda_margin=0.14, revenue_growth=0.04, da_pct=0.035, capex_pct=0.03,
        nwc_pct=0.06, tax_rate=0.24, debt_rate=0.095, mandatory_amort_pct=0.05,
        cash_sweep_pct=0.85, fees_pct=0.02, min_cash=0.0,
    ),
}
