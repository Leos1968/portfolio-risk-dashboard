"""Offline unit tests for the IB/PE modeling engines.

These are deterministic (no market data) and enforce the properties that make a
model trustworthy: the balance sheet balances, the statements articulate, and
the M&A / LBO arithmetic matches hand-computed examples.
"""

import numpy as np
import pytest

from modeling import lbo_model as lbo
from modeling import ma_model as ma
from modeling import three_statement as ts


# ===========================================================================
# Three-statement model
# ===========================================================================

def _base() -> ts.BaseFinancials:
    return ts.BaseFinancials(
        revenue=1000.0, cogs=600.0, sga=200.0, rd=0.0, da=50.0,
        cash=80.0, accounts_receivable=120.0, inventory=100.0, net_ppe=500.0,
        other_assets=40.0, accounts_payable=90.0, debt=400.0, revolver=0.0,
        other_liabilities=60.0, common_stock=150.0, retained_earnings=40.0,
    )


def _assumptions(**over) -> ts.Assumptions:
    base = dict(
        years=5, revenue_growth=0.10, gross_margin=0.40, sga_pct=0.18, da_pct=0.05,
        capex_pct=0.06, tax_rate=0.25, dso=45, dio=60, dpo=40,
        interest_rate=0.06, revolver_rate=0.08, cash_yield=0.02, min_cash=50.0,
        mandatory_amort=20.0, cash_sweep_pct=0.0, payout_ratio=0.20,
        equity_issuance=0.0, use_revolver=True,
    )
    base.update(over)
    return ts.Assumptions(**base)


def test_balance_sheet_balances_every_year():
    model = ts.build_model(_base(), _assumptions())
    assert model.balances
    assert model.max_balance_error < 1e-6
    check_row = model.balance_sheet.loc["— Check: Assets − (L+E)"]
    assert np.allclose(check_row.to_numpy(dtype=float), 0.0, atol=1e-6)


def test_cfs_ending_cash_equals_balance_sheet_cash():
    model = ts.build_model(_base(), _assumptions())
    bs_cash = model.balance_sheet.loc["Cash & Equivalents"].to_numpy(dtype=float)
    cfs_cash = model.cash_flow.loc["— Ending Cash"].to_numpy(dtype=float)
    assert np.allclose(bs_cash, cfs_cash, atol=1e-9)


def test_net_change_in_cash_reconciles():
    model = ts.build_model(_base(), _assumptions())
    cash = model.balance_sheet.loc["Cash & Equivalents"].to_numpy(dtype=float)
    net_change = model.cash_flow.loc["Net Change in Cash"].to_numpy(dtype=float)
    assert np.allclose(np.diff(cash), net_change[1:], atol=1e-9)


def test_retained_earnings_rolls_with_net_income_less_dividends():
    model = ts.build_model(_base(), _assumptions())
    re = model.balance_sheet.loc["Retained Earnings"].to_numpy(dtype=float)
    ni = model.income_statement.loc["Net Income"].to_numpy(dtype=float)
    div = -model.cash_flow.loc["Dividends Paid"].to_numpy(dtype=float)
    assert np.allclose(np.diff(re), (ni - div)[1:], atol=1e-9)


def test_balances_under_stress_high_growth_and_working_capital():
    a = _assumptions(revenue_growth=0.40, dso=90, dio=120, dpo=20,
                     capex_pct=0.15, payout_ratio=0.5, cash_sweep_pct=0.5)
    model = ts.build_model(_base(), a)
    assert model.balances


def test_balances_without_revolver_even_if_cash_goes_negative():
    a = _assumptions(use_revolver=False, min_cash=0.0, capex_pct=0.30,
                     revenue_growth=0.5, payout_ratio=0.8)
    model = ts.build_model(_base(), a)
    assert model.balances  # still articulates; cash may be negative and that's fine


def test_revolver_holds_minimum_cash():
    a = _assumptions(min_cash=50.0, capex_pct=0.25, revenue_growth=0.4, cash_yield=0.0)
    model = ts.build_model(_base(), a)
    cash = model.balance_sheet.loc["Cash & Equivalents"].to_numpy(dtype=float)
    assert cash.min() >= 50.0 - 1e-6


def test_equity_issuance_flows_to_common_stock_and_cash():
    a = _assumptions(equity_issuance=100.0, use_revolver=True)
    model = ts.build_model(_base(), a)
    common = model.balance_sheet.loc["Common Stock / APIC"].to_numpy(dtype=float)
    assert common[1] == pytest.approx(common[0] + 100.0)
    assert model.balances


# ===========================================================================
# M&A — accretion / dilution
# ===========================================================================

def test_all_stock_accretion_matches_hand_calc():
    acq = ma.Acquirer(net_income=1000, shares=1000, share_price=20, tax_rate=0.25)  # EPS 1.0, P/E 20
    tgt = ma.Target(net_income=100, shares=200, share_price=8)                      # P/E paid 16 @ 0 premium
    terms = ma.DealTerms(offer_premium=0.0, pct_stock=1.0, pct_cash=0.0, pct_debt=0.0)
    r = ma.run_deal(acq, tgt, terms)
    assert r.new_shares == pytest.approx(80.0)          # $1,600 / $20
    assert r.proforma_eps == pytest.approx(1100 / 1080)
    assert r.accretion_dilution == pytest.approx(1100 / 1080 - 1.0)
    assert r.accretion_dilution > 0                     # P/E 20 > 16 → accretive


def test_all_debt_deal_reflects_after_tax_interest():
    acq = ma.Acquirer(net_income=1000, shares=1000, share_price=10, tax_rate=0.25)
    tgt = ma.Target(net_income=100, shares=100, share_price=16)
    terms = ma.DealTerms(offer_premium=0.0, pct_stock=0.0, pct_cash=0.0, pct_debt=1.0,
                         new_debt_rate=0.05)
    r = ma.run_deal(acq, tgt, terms)
    # Purchase $1,600 of debt @5% × (1−25%) = $60 after-tax interest
    assert r.new_shares == pytest.approx(0.0)
    assert r.proforma_net_income == pytest.approx(1040.0)
    assert r.proforma_eps == pytest.approx(1.04)
    assert r.accretion_dilution == pytest.approx(0.04)


def test_breakeven_synergies_make_deal_neutral():
    acq = ma.Acquirer(net_income=1000, shares=1000, share_price=10, tax_rate=0.25)  # P/E 10
    tgt = ma.Target(net_income=100, shares=100, share_price=20)                     # P/E paid 20
    terms = ma.DealTerms(offer_premium=0.0, pct_stock=1.0, pct_cash=0.0, pct_debt=0.0)
    r = ma.run_deal(acq, tgt, terms)
    assert r.accretion_dilution < 0                     # buying a pricier P/E with stock → dilutive
    neutral = ma.run_deal(acq, tgt, ma.DealTerms(
        offer_premium=0.0, pct_stock=1.0, pct_cash=0.0, pct_debt=0.0,
        pretax_synergies=r.breakeven_pretax_synergies))
    assert neutral.accretion_dilution == pytest.approx(0.0, abs=1e-9)


def test_sources_equal_uses():
    acq = ma.Acquirer(net_income=1000, shares=1000, share_price=20)
    tgt = ma.Target(net_income=100, shares=200, share_price=8, net_debt=150, book_equity=300)
    terms = ma.DealTerms(offer_premium=0.30, pct_stock=0.5, pct_cash=0.25, pct_debt=0.25,
                         transaction_fees=25.0)
    r = ma.run_deal(acq, tgt, terms)
    su = r.sources_uses
    total_sources = su["Sources"].iloc[-1][1]
    total_uses = su["Uses"].iloc[-1][1]
    assert total_sources == pytest.approx(total_uses)
    assert r.stock_consideration + r.cash_consideration + r.debt_consideration == pytest.approx(total_uses)


def test_ownership_split_sums_to_one():
    acq = ma.Acquirer(net_income=1000, shares=1000, share_price=20)
    tgt = ma.Target(net_income=100, shares=200, share_price=8)
    r = ma.run_deal(acq, tgt, ma.DealTerms(offer_premium=0.2, pct_stock=0.6, pct_cash=0.4, pct_debt=0.0))
    assert r.acquirer_ownership + r.target_ownership == pytest.approx(1.0)


# ===========================================================================
# LBO — returns
# ===========================================================================

def _lbo() -> lbo.LBOAssumptions:
    return lbo.LBOAssumptions(
        entry_revenue=1000.0, entry_multiple=10.0, exit_multiple=10.0, entry_leverage=5.0,
        years=5, ebitda_margin=0.20, revenue_growth=0.05, da_pct=0.05, capex_pct=0.04,
        nwc_pct=0.10, tax_rate=0.25, debt_rate=0.08, mandatory_amort_pct=0.05,
        cash_sweep_pct=0.80, fees_pct=0.02, min_cash=0.0,
    )


def test_lbo_moic_and_irr_are_consistent():
    r = lbo.run_lbo(_lbo())
    assert r.moic == pytest.approx(r.exit_equity / r.sponsor_equity)
    assert r.irr == pytest.approx(r.moic ** (1 / 5) - 1)


def test_lbo_value_bridge_reconciles_to_total_gain():
    r = lbo.run_lbo(_lbo())
    parts = r.bridge["EBITDA growth"] + r.bridge["Multiple change"] + r.bridge["Debt paydown"] + r.bridge["Fees"]
    assert parts == pytest.approx(r.bridge["Total equity gain"], abs=1e-6)


def test_lbo_deleverages_when_sweeping_positive_fcf():
    r = lbo.run_lbo(_lbo())
    net_debt = r.schedule.loc["Net Debt (end)"].to_numpy(dtype=float)
    assert net_debt[-1] < net_debt[0]            # debt falls over the hold
    assert r.exit_leverage < r.entry_leverage


def test_lbo_higher_exit_multiple_raises_irr():
    low = lbo.run_lbo(_lbo())
    hi_assumptions = _lbo()
    hi_assumptions.exit_multiple = 12.0
    hi = lbo.run_lbo(hi_assumptions)
    assert hi.irr > low.irr


def test_lbo_sources_equal_uses():
    a = _lbo()
    r = lbo.run_lbo(a)
    uses = r.enterprise_value + r.fees + a.min_cash
    sources = r.new_debt + r.sponsor_equity
    assert sources == pytest.approx(uses)
