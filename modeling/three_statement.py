"""
Three-Statement Model
=====================
A driver-based, fully-linked income statement, balance sheet, and cash-flow
statement — the workhorse of every banking and buy-side desk.

The three statements *articulate*: net income flows to retained earnings on the
balance sheet and to the top of the cash-flow statement; depreciation is added
back on the CFS and rolls net PP&E on the BS; working-capital movements hit both
the CFS and the BS; and the ending cash balance on the CFS *is* the cash line on
the BS. Build the CFS this way and the balance sheet balances **by construction**
— Assets ≡ Liabilities + Equity, to the penny, in every projected year.

Proof of articulation (per year):
    Δcash (CFS)  = NI + D&A − ΔAR − ΔInv + ΔAP − capex + Δdebt − dividends + Δequity
    ΔAssets      = Δcash + ΔAR + ΔInv + (capex − D&A)
    Δ(L+E)       = ΔAP + Δdebt + (NI − dividends) + Δequity
    Substituting Δcash into ΔAssets collapses it to Δ(L+E). ∎

Design choices (all standard for a clean, transparent model):
  * Interest is charged on *beginning-of-period* debt, so the model has no
    circular reference (interest → net income → cash → debt → interest) and needs
    no iterative solver. Documented as a caveat in the app.
  * A revolver is the cash plug: it is drawn to hold a minimum cash balance and
    swept down with surplus cash. An optional term-loan cash sweep prepays debt
    with excess free cash flow.
  * Taxes are floored at zero (no immediate refund on pre-tax losses) — the
    conservative convention; NOLs are out of scope.
  * "Other assets" and "other liabilities" are held flat, so the base-year
    balance is preserved forward. Any base-year imbalance is absorbed into a
    labelled "Other assets (plug)" so the model always starts balanced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS = 365.0


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class BaseFinancials:
    """Year-0 actuals: the jumping-off point for the projection.

    Income-statement items are full-year figures; balance-sheet items are
    period-end. All values in the same currency unit (e.g. $ millions).
    """

    revenue: float
    # Income statement (year 0, informational + continuity)
    cogs: float
    sga: float
    rd: float = 0.0
    da: float = 0.0
    # Balance sheet (period end)
    cash: float = 0.0
    accounts_receivable: float = 0.0
    inventory: float = 0.0
    net_ppe: float = 0.0
    other_assets: float = 0.0
    accounts_payable: float = 0.0
    debt: float = 0.0            # total interest-bearing term debt at t0
    revolver: float = 0.0        # drawn revolver at t0 (usually 0)
    other_liabilities: float = 0.0
    common_stock: float = 0.0    # paid-in capital
    retained_earnings: float = 0.0


@dataclass
class Assumptions:
    """Forward drivers. Scalars are held constant; lists set per-year paths.

    Ratios are decimals (0.08 = 8%). Working-capital drivers are in days.
    """

    years: int = 5
    revenue_growth: float | list[float] = 0.08
    gross_margin: float | list[float] = 0.40      # gross profit / revenue
    sga_pct: float | list[float] = 0.20           # SG&A / revenue
    rd_pct: float | list[float] = 0.0             # R&D / revenue
    da_pct: float | list[float] = 0.04            # D&A / revenue
    capex_pct: float | list[float] = 0.05         # capex / revenue
    tax_rate: float | list[float] = 0.23
    dso: float | list[float] = 45.0               # days sales outstanding  → AR
    dio: float | list[float] = 60.0               # days inventory on hand  → Inventory
    dpo: float | list[float] = 40.0               # days payable outstanding → AP
    interest_rate: float | list[float] = 0.06     # on beginning term debt
    revolver_rate: float | list[float] = 0.08     # on beginning revolver
    cash_yield: float | list[float] = 0.03        # interest income on beginning cash
    min_cash: float = 0.0                         # revolver holds cash at/above this
    mandatory_amort: float | list[float] = 0.0    # scheduled term repayment ($/yr)
    cash_sweep_pct: float | list[float] = 0.0     # % of surplus FCF prepaying term
    payout_ratio: float | list[float] = 0.0       # dividends / net income
    equity_issuance: float | list[float] = 0.0    # net new equity raised ($/yr)
    use_revolver: bool = True

    def path(self, name: str) -> list[float]:
        """Broadcast a scalar assumption to a per-year list of length `years`."""
        value = getattr(self, name)
        if isinstance(value, (list, tuple, np.ndarray)):
            seq = [float(v) for v in value]
            if len(seq) < self.years:
                seq = seq + [seq[-1]] * (self.years - len(seq))
            return seq[: self.years]
        return [float(value)] * self.years


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ThreeStatementModel:
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    metrics: pd.DataFrame
    max_balance_error: float          # largest |Assets − (L+E)| across all years
    balances: bool                    # within tolerance?
    columns: list[str]                # ["Year 0", "Year 1", ...]

    def rounded(self, decimals: int = 1) -> "ThreeStatementModel":
        return ThreeStatementModel(
            self.income_statement.round(decimals),
            self.balance_sheet.round(decimals),
            self.cash_flow.round(decimals),
            self.metrics.round(decimals),
            self.max_balance_error,
            self.balances,
            self.columns,
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def build_model(base: BaseFinancials, a: Assumptions) -> ThreeStatementModel:
    """Project the three linked statements `a.years` years past the base year."""
    n = a.years
    yrs = range(1, n + 1)

    g = a.path("revenue_growth")
    gm = a.path("gross_margin")
    sga_p = a.path("sga_pct")
    rd_p = a.path("rd_pct")
    da_p = a.path("da_pct")
    capex_p = a.path("capex_pct")
    tax = a.path("tax_rate")
    dso, dio, dpo = a.path("dso"), a.path("dio"), a.path("dpo")
    ir, rr, cy = a.path("interest_rate"), a.path("revolver_rate"), a.path("cash_yield")
    amort_s = a.path("mandatory_amort")
    sweep_s = a.path("cash_sweep_pct")
    payout = a.path("payout_ratio")
    eq_iss = a.path("equity_issuance")

    # Base-year plug so the opening balance sheet is exactly balanced.
    assets0 = base.cash + base.accounts_receivable + base.inventory + base.net_ppe + base.other_assets
    le0 = (base.accounts_payable + base.debt + base.revolver + base.other_liabilities
           + base.common_stock + base.retained_earnings)
    other_assets_plug = base.other_assets + (le0 - assets0)  # forces Assets == L+E at t0

    # Column-0 (base year) records --------------------------------------------
    rev = [base.revenue]
    cogs = [base.cogs]
    gross = [base.revenue - base.cogs]
    sga = [base.sga]
    rd = [base.rd]
    da = [base.da]
    ebit = [gross[0] - base.sga - base.rd - base.da]
    ebitda = [ebit[0] + base.da]
    int_exp = [0.0]
    int_inc = [0.0]
    ebt = [ebit[0]]
    taxes = [0.0]
    ni = [ebit[0]]

    cash = [base.cash]
    ar = [base.accounts_receivable]
    inv = [base.inventory]
    ppe = [base.net_ppe]
    term = [base.debt]
    revolver = [base.revolver]
    ap = [base.accounts_payable]
    retained = [base.retained_earnings]
    common = [base.common_stock]

    capex = [0.0]
    dividends = [0.0]
    cfo = [0.0]
    cfi = [0.0]
    cff = [0.0]
    d_cash = [0.0]

    for i, t in enumerate(yrs):
        # --- Income statement (interest on BEGINNING balances) ---------------
        revenue = rev[t - 1] * (1.0 + g[i])
        cost = revenue * (1.0 - gm[i])
        gp = revenue - cost
        sga_t = revenue * sga_p[i]
        rd_t = revenue * rd_p[i]
        da_t = revenue * da_p[i]
        ebit_t = gp - sga_t - rd_t - da_t

        interest = ir[i] * term[t - 1] + rr[i] * revolver[t - 1]
        income_cash = cy[i] * cash[t - 1]
        ebt_t = ebit_t - interest + income_cash
        tax_t = tax[i] * max(ebt_t, 0.0)
        ni_t = ebt_t - tax_t

        # --- Balance-sheet drivers (working capital + PP&E) ------------------
        ar_t = revenue / DAYS * dso[i]
        inv_t = cost / DAYS * dio[i]
        ap_t = cost / DAYS * dpo[i]
        capex_t = revenue * capex_p[i]
        ppe_t = ppe[t - 1] + capex_t - da_t

        d_ar = ar_t - ar[t - 1]
        d_inv = inv_t - inv[t - 1]
        d_ap = ap_t - ap[t - 1]

        # --- Cash flow (pre-financing) ---------------------------------------
        cfo_t = ni_t + da_t - d_ar - d_inv + d_ap
        cfi_t = -capex_t
        div_t = payout[i] * max(ni_t, 0.0)
        eq_t = eq_iss[i]

        cash_avail = cash[t - 1] + cfo_t + cfi_t - div_t + eq_t

        # --- Debt schedule: mandatory amort, revolver plug, optional sweep ---
        amort = min(amort_s[i], term[t - 1])
        term_t = term[t - 1] - amort
        cash_avail -= amort

        rev_begin = revolver[t - 1]
        draw = repay = 0.0
        if a.use_revolver:
            if cash_avail < a.min_cash:
                draw = a.min_cash - cash_avail
                cash_avail = a.min_cash
            else:
                repay = min(cash_avail - a.min_cash, rev_begin)
                cash_avail -= repay
        revolver_t = rev_begin + draw - repay

        sweep = 0.0
        if sweep_s[i] > 0 and cash_avail > a.min_cash and term_t > 0:
            sweep = min(sweep_s[i] * (cash_avail - a.min_cash), term_t)
            term_t -= sweep
            cash_avail -= sweep

        cash_t = cash_avail
        cff_t = (draw - repay - amort - sweep) + (-div_t) + eq_t
        d_cash_t = cfo_t + cfi_t + cff_t

        # --- Equity roll -----------------------------------------------------
        retained_t = retained[t - 1] + ni_t - div_t
        common_t = common[t - 1] + eq_t

        # --- Store -----------------------------------------------------------
        rev.append(revenue); cogs.append(cost); gross.append(gp)
        sga.append(sga_t); rd.append(rd_t); da.append(da_t)
        ebit.append(ebit_t); ebitda.append(ebit_t + da_t)
        int_exp.append(interest); int_inc.append(income_cash)
        ebt.append(ebt_t); taxes.append(tax_t); ni.append(ni_t)
        cash.append(cash_t); ar.append(ar_t); inv.append(inv_t); ppe.append(ppe_t)
        term.append(term_t); revolver.append(revolver_t); ap.append(ap_t)
        retained.append(retained_t); common.append(common_t)
        capex.append(capex_t); dividends.append(div_t)
        cfo.append(cfo_t); cfi.append(cfi_t); cff.append(cff_t); d_cash.append(d_cash_t)

    cols = ["Year 0"] + [f"Year {t}" for t in yrs]
    other_assets_row = [other_assets_plug] * (n + 1)
    other_liab_row = [base.other_liabilities] * (n + 1)

    income_statement = _frame(cols, [
        ("Revenue", rev),
        ("COGS", [-c for c in cogs]),
        ("Gross Profit", gross),
        ("SG&A", [-s for s in sga]),
        ("R&D", [-r for r in rd]),
        ("Depreciation & Amortization", [-d for d in da]),
        ("EBIT", ebit),
        ("Interest Expense", [-x for x in int_exp]),
        ("Interest Income", int_inc),
        ("Pre-Tax Income (EBT)", ebt),
        ("Taxes", [-x for x in taxes]),
        ("Net Income", ni),
        ("— Memo: EBITDA", ebitda),
    ])

    total_assets = [cash[i] + ar[i] + inv[i] + ppe[i] + other_assets_row[i] for i in range(n + 1)]
    total_le = [ap[i] + term[i] + revolver[i] + other_liab_row[i] + common[i] + retained[i]
                for i in range(n + 1)]
    balance_check = [total_assets[i] - total_le[i] for i in range(n + 1)]

    balance_sheet = _frame(cols, [
        ("Cash & Equivalents", cash),
        ("Accounts Receivable", ar),
        ("Inventory", inv),
        ("Net PP&E", ppe),
        ("Other Assets", other_assets_row),
        ("Total Assets", total_assets),
        ("Accounts Payable", ap),
        ("Revolver", revolver),
        ("Term Debt", term),
        ("Other Liabilities", other_liab_row),
        ("Common Stock / APIC", common),
        ("Retained Earnings", retained),
        ("Total Liabilities & Equity", total_le),
        ("— Check: Assets − (L+E)", balance_check),
    ])

    cash_flow = _frame(cols, [
        ("Net Income", ni),
        ("D&A", da),
        ("Δ Accounts Receivable", [0.0] + [-(ar[i] - ar[i - 1]) for i in range(1, n + 1)]),
        ("Δ Inventory", [0.0] + [-(inv[i] - inv[i - 1]) for i in range(1, n + 1)]),
        ("Δ Accounts Payable", [0.0] + [(ap[i] - ap[i - 1]) for i in range(1, n + 1)]),
        ("Cash Flow from Operations", cfo),
        ("Capital Expenditures", [-c for c in capex]),
        ("Cash Flow from Investing", cfi),
        ("Δ Revolver", [0.0] + [(revolver[i] - revolver[i - 1]) for i in range(1, n + 1)]),
        ("Δ Term Debt", [0.0] + [(term[i] - term[i - 1]) for i in range(1, n + 1)]),
        ("Dividends Paid", [-d for d in dividends]),
        ("Equity Issued", [0.0] + [common[i] - common[i - 1] for i in range(1, n + 1)]),
        ("Cash Flow from Financing", cff),
        ("Net Change in Cash", d_cash),
        ("— Ending Cash", cash),
    ])

    net_debt = [term[i] + revolver[i] - cash[i] for i in range(n + 1)]
    fcf = [cfo[i] + cfi[i] for i in range(n + 1)]
    metrics = _frame(cols, [
        ("Revenue Growth %", [np.nan] + [(rev[i] / rev[i - 1] - 1.0) * 100 for i in range(1, n + 1)]),
        ("Gross Margin %", [gross[i] / rev[i] * 100 if rev[i] else np.nan for i in range(n + 1)]),
        ("EBITDA Margin %", [ebitda[i] / rev[i] * 100 if rev[i] else np.nan for i in range(n + 1)]),
        ("EBIT Margin %", [ebit[i] / rev[i] * 100 if rev[i] else np.nan for i in range(n + 1)]),
        ("Net Margin %", [ni[i] / rev[i] * 100 if rev[i] else np.nan for i in range(n + 1)]),
        ("Free Cash Flow", fcf),
        ("Net Debt", net_debt),
        ("Net Debt / EBITDA", [net_debt[i] / ebitda[i] if ebitda[i] else np.nan for i in range(n + 1)]),
        ("Interest Coverage (EBIT/Int)", [ebit[i] / int_exp[i] if int_exp[i] else np.nan for i in range(n + 1)]),
    ])

    max_err = float(np.max(np.abs(balance_check)))
    scale = max(abs(base.revenue), 1.0)
    return ThreeStatementModel(
        income_statement, balance_sheet, cash_flow, metrics,
        max_balance_error=max_err,
        balances=max_err < 1e-6 * scale + 1e-6,
        columns=cols,
    )


def _frame(cols: list[str], rows: list[tuple[str, list[float]]]) -> pd.DataFrame:
    return pd.DataFrame({label: values for label, values in rows}, index=cols).T
