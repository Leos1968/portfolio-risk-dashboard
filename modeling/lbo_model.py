"""
LBO — Leveraged Buyout Returns
==============================
The private-equity lens: buy a business with mostly borrowed money, use its own
cash flow to pay the debt down, grow EBITDA, and sell in ~5 years. Returns come
from three levers — and a good memo says which one it is leaning on:

    ΔEquity value ≈  (EBITDA growth × entry multiple)          ← operations
                   + (multiple change × exit EBITDA)            ← re-rating
                   + (net-debt paydown from free cash flow)     ← deleveraging

Headline metrics:
    MOIC = exit equity / sponsor equity              (multiple of invested capital)
    IRR  = MOIC**(1/years) − 1                        (no interim distributions)

Sources & uses at entry:
    Uses    = enterprise value + fees + minimum cash
    Sources = new debt (entry_leverage × entry EBITDA) + sponsor equity

As in the three-statement model, interest is charged on *beginning-of-period*
debt to keep the model free of circular references. Free cash flow first meets
mandatory amortization, then an optional cash sweep prepays debt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LBOAssumptions:
    entry_revenue: float
    entry_multiple: float           # entry EV / EBITDA
    exit_multiple: float            # exit EV / EBITDA
    entry_leverage: float           # opening net debt / EBITDA → sizes the debt
    years: int = 5
    ebitda_margin: float | list[float] = 0.25
    revenue_growth: float | list[float] = 0.06
    da_pct: float | list[float] = 0.05          # D&A / revenue
    capex_pct: float | list[float] = 0.04       # capex / revenue
    nwc_pct: float = 0.10                        # net working capital / revenue
    tax_rate: float = 0.23
    debt_rate: float = 0.09                      # blended cash interest on beginning debt
    mandatory_amort_pct: float = 0.05            # % of initial debt repaid each year
    cash_sweep_pct: float = 0.80                 # % of surplus FCF prepaying debt
    fees_pct: float = 0.03                       # transaction + financing fees, % of EV
    min_cash: float = 0.0

    def path(self, name: str) -> list[float]:
        value = getattr(self, name)
        if isinstance(value, (list, tuple, np.ndarray)):
            seq = [float(v) for v in value]
            if len(seq) < self.years:
                seq = seq + [seq[-1]] * (self.years - len(seq))
            return seq[: self.years]
        return [float(value)] * self.years


@dataclass
class LBOResult:
    enterprise_value: float
    new_debt: float
    sponsor_equity: float
    fees: float
    entry_ebitda: float
    exit_ebitda: float
    exit_ev: float
    exit_net_debt: float
    exit_equity: float
    moic: float
    irr: float
    entry_leverage: float
    exit_leverage: float
    schedule: pd.DataFrame
    bridge: dict[str, float]        # value-creation attribution → total gain


def run_lbo(a: LBOAssumptions) -> LBOResult:
    """Run the buyout: entry structure → debt paydown → exit → IRR/MOIC."""
    n = a.years
    margin = a.path("ebitda_margin")
    g = a.path("revenue_growth")
    da_p = a.path("da_pct")
    capex_p = a.path("capex_pct")

    entry_ebitda = a.entry_revenue * margin[0]
    ev = entry_ebitda * a.entry_multiple
    new_debt = a.entry_leverage * entry_ebitda
    fees = a.fees_pct * ev
    sponsor_equity = ev + fees + a.min_cash - new_debt

    rev = a.entry_revenue
    debt = new_debt
    cash = a.min_cash
    rows = []

    for i in range(n):
        rev_prev = rev
        rev = rev_prev * (1.0 + g[i])
        ebitda = rev * margin[i]
        da = rev * da_p[i]
        ebit = ebitda - da
        interest = a.debt_rate * debt                 # beginning debt
        ebt = ebit - interest
        tax = a.tax_rate * max(ebt, 0.0)
        capex = rev * capex_p[i]
        d_nwc = a.nwc_pct * (rev - rev_prev)           # growth ties up working capital
        fcf = ebitda - interest - tax - capex - d_nwc  # free cash flow before debt paydown

        amort = min(a.mandatory_amort_pct * new_debt, debt)
        cash_after = cash + fcf - amort
        debt_after = debt - amort

        sweep = 0.0
        if a.cash_sweep_pct > 0 and cash_after > a.min_cash and debt_after > 0:
            sweep = min(a.cash_sweep_pct * (cash_after - a.min_cash), debt_after)
            debt_after -= sweep
            cash_after -= sweep

        debt, cash = debt_after, cash_after
        rows.append({
            "Year": i + 1,
            "Revenue": rev,
            "EBITDA": ebitda,
            "EBIT": ebit,
            "Interest": interest,
            "Taxes": tax,
            "Capex": capex,
            "Δ NWC": d_nwc,
            "Free Cash Flow": fcf,
            "Mandatory Amort": amort,
            "Cash Sweep": sweep,
            "Debt (end)": debt,
            "Cash (end)": cash,
            "Net Debt (end)": debt - cash,
            "Net Debt / EBITDA": (debt - cash) / ebitda if ebitda else np.nan,
        })

    schedule = pd.DataFrame(rows).set_index("Year").T

    exit_ebitda = rows[-1]["EBITDA"]
    exit_ev = exit_ebitda * a.exit_multiple
    exit_net_debt = rows[-1]["Net Debt (end)"]
    exit_equity = exit_ev - exit_net_debt
    moic = exit_equity / sponsor_equity if sponsor_equity > 0 else float("nan")
    irr = moic ** (1.0 / n) - 1.0 if (moic and moic > 0) else float("nan")

    entry_net_debt = new_debt - a.min_cash
    bridge = {
        "EBITDA growth": (exit_ebitda - entry_ebitda) * a.entry_multiple,
        "Multiple change": (a.exit_multiple - a.entry_multiple) * exit_ebitda,
        "Debt paydown": entry_net_debt - exit_net_debt,
        "Fees": -fees,
        "Total equity gain": exit_equity - sponsor_equity,
    }

    return LBOResult(
        enterprise_value=ev, new_debt=new_debt, sponsor_equity=sponsor_equity, fees=fees,
        entry_ebitda=entry_ebitda, exit_ebitda=exit_ebitda, exit_ev=exit_ev,
        exit_net_debt=exit_net_debt, exit_equity=exit_equity, moic=moic, irr=irr,
        entry_leverage=entry_net_debt / entry_ebitda if entry_ebitda else float("nan"),
        exit_leverage=exit_net_debt / exit_ebitda if exit_ebitda else float("nan"),
        schedule=schedule, bridge=bridge,
    )


def irr_sensitivity(a: LBOAssumptions, exit_multiples: list[float],
                    entry_leverages: list[float]) -> pd.DataFrame:
    """IRR (%) across an exit-multiple × entry-leverage grid.

    Rows = exit multiple, columns = entry leverage (turns of EBITDA).
    """
    out = np.zeros((len(exit_multiples), len(entry_leverages)))
    for i, xm in enumerate(exit_multiples):
        for j, lev in enumerate(entry_leverages):
            trial = LBOAssumptions(
                entry_revenue=a.entry_revenue, entry_multiple=a.entry_multiple,
                exit_multiple=xm, entry_leverage=lev, years=a.years,
                ebitda_margin=a.ebitda_margin, revenue_growth=a.revenue_growth,
                da_pct=a.da_pct, capex_pct=a.capex_pct, nwc_pct=a.nwc_pct,
                tax_rate=a.tax_rate, debt_rate=a.debt_rate,
                mandatory_amort_pct=a.mandatory_amort_pct, cash_sweep_pct=a.cash_sweep_pct,
                fees_pct=a.fees_pct, min_cash=a.min_cash,
            )
            out[i, j] = run_lbo(trial).irr * 100.0
    return pd.DataFrame(out, index=[f"{m:.1f}x" for m in exit_multiples],
                        columns=[f"{l:.1f}x" for l in entry_leverages])
