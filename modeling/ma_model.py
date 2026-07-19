"""
M&A — Accretion / Dilution
==========================
The first question a banker answers on any acquisition: *does this deal add to
or subtract from the acquirer's earnings per share?* The mechanics:

    Pro-forma net income = Acquirer NI + Target NI
                           + after-tax synergies
                           − after-tax interest on new acquisition debt
                           − after-tax foregone interest on cash used
    Pro-forma shares      = Acquirer shares + new shares issued for stock
    Pro-forma EPS         = Pro-forma NI / Pro-forma shares
    Accretion/(Dilution)  = Pro-forma EPS / Standalone acquirer EPS − 1

Intuition (the P/E rule of thumb): paying with a *high-P/E* currency (cheap
stock) or with financing whose after-tax cost is below the target's earnings
yield is accretive; the reverse is dilutive. An all-stock deal is accretive
whenever the acquirer's P/E exceeds the P/E paid for the target.

Everything here is pure arithmetic on forward earnings estimates — no market
data, no I/O — so it is deterministic and unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Acquirer:
    net_income: float          # forward (NTM) net income
    shares: float              # diluted shares outstanding
    share_price: float
    tax_rate: float = 0.23
    net_debt: float = 0.0      # for pro-forma leverage
    ebitda: float = 0.0        # for pro-forma leverage

    @property
    def eps(self) -> float:
        return self.net_income / self.shares if self.shares else float("nan")

    @property
    def pe(self) -> float:
        return self.share_price / self.eps if self.eps else float("nan")


@dataclass
class Target:
    net_income: float          # forward (NTM) net income
    shares: float
    share_price: float         # unaffected (pre-announcement) price
    net_debt: float = 0.0      # assumed/refinanced in the deal
    ebitda: float = 0.0
    book_equity: float = 0.0   # for goodwill

    @property
    def eps(self) -> float:
        return self.net_income / self.shares if self.shares else float("nan")


@dataclass
class DealTerms:
    offer_premium: float = 0.30      # premium to target's unaffected price
    pct_stock: float = 0.50          # consideration mix (must sum to 1 with cash+debt)
    pct_cash: float = 0.25
    pct_debt: float = 0.25
    new_debt_rate: float = 0.06      # interest on acquisition debt
    cash_yield: float = 0.03         # foregone yield on the acquirer's cash used
    pretax_synergies: float = 0.0    # annual run-rate cost/revenue synergies
    assume_target_debt: bool = True  # refinance target net debt as a use of funds
    transaction_fees: float = 0.0    # advisory/financing fees (funded pro-rata)


@dataclass
class DealResult:
    offer_price: float
    equity_purchase_price: float
    enterprise_value: float
    new_shares: float
    stock_consideration: float
    cash_consideration: float
    debt_consideration: float
    standalone_eps: float
    proforma_eps: float
    accretion_dilution: float          # decimal, +accretive / −dilutive
    proforma_net_income: float
    proforma_shares: float
    goodwill: float
    acquirer_ownership: float
    target_ownership: float
    proforma_net_debt: float
    proforma_leverage: float           # PF net debt / PF EBITDA (nan if no EBITDA)
    breakeven_pretax_synergies: float  # synergies that make EPS-neutral
    sources_uses: pd.DataFrame
    verdict: str


def run_deal(acq: Acquirer, tgt: Target, terms: DealTerms) -> DealResult:
    """Compute the full accretion/dilution bridge for one deal."""
    offer_price = tgt.share_price * (1.0 + terms.offer_premium)
    equity_price = offer_price * tgt.shares
    target_debt = tgt.net_debt if terms.assume_target_debt else 0.0
    total_uses = equity_price + target_debt + terms.transaction_fees
    enterprise_value = equity_price + tgt.net_debt

    # Consideration mix funds the *equity* purchase; fees + refinanced debt are
    # funded with incremental acquisition debt/cash pro-rata to the cash+debt mix.
    stock_consid = equity_price * terms.pct_stock
    non_equity_uses = total_uses - stock_consid
    cash_debt = terms.pct_cash + terms.pct_debt
    if cash_debt > 0:
        cash_consid = non_equity_uses * (terms.pct_cash / cash_debt)
        debt_consid = non_equity_uses * (terms.pct_debt / cash_debt)
    else:
        cash_consid = debt_consid = 0.0

    new_shares = stock_consid / acq.share_price if acq.share_price else 0.0

    after_tax_syn = terms.pretax_synergies * (1.0 - acq.tax_rate)
    after_tax_new_int = debt_consid * terms.new_debt_rate * (1.0 - acq.tax_rate)
    after_tax_foregone = cash_consid * terms.cash_yield * (1.0 - acq.tax_rate)

    pf_ni = (acq.net_income + tgt.net_income + after_tax_syn
             - after_tax_new_int - after_tax_foregone)
    pf_shares = acq.shares + new_shares
    pf_eps = pf_ni / pf_shares if pf_shares else float("nan")
    accretion = pf_eps / acq.eps - 1.0 if acq.eps else float("nan")

    goodwill = equity_price - tgt.book_equity
    acq_own = acq.shares / pf_shares if pf_shares else float("nan")
    tgt_own = new_shares / pf_shares if pf_shares else float("nan")

    pf_net_debt = acq.net_debt + debt_consid + cash_consid + target_debt
    pf_ebitda = acq.ebitda + tgt.ebitda
    pf_leverage = pf_net_debt / pf_ebitda if pf_ebitda else float("nan")

    breakeven_syn = _breakeven_synergies(acq, tgt, terms, new_shares,
                                         after_tax_new_int, after_tax_foregone)

    sources_uses = pd.DataFrame(
        {
            "Sources": [
                ("New equity issued", stock_consid),
                ("New acquisition debt", debt_consid),
                ("Cash on hand", cash_consid),
                ("Total sources", stock_consid + debt_consid + cash_consid),
            ],
            "Uses": [
                ("Equity purchase price", equity_price),
                ("Refinance target net debt", target_debt),
                ("Transaction fees", terms.transaction_fees),
                ("Total uses", total_uses),
            ],
        }
    )

    tag = "accretive" if accretion > 1e-9 else ("dilutive" if accretion < -1e-9 else "neutral")
    verdict = f"{accretion:+.1%} — {tag}"

    return DealResult(
        offer_price, equity_price, enterprise_value, new_shares,
        stock_consid, cash_consid, debt_consid,
        acq.eps, pf_eps, accretion, pf_ni, pf_shares, goodwill,
        acq_own, tgt_own, pf_net_debt, pf_leverage, breakeven_syn,
        sources_uses, verdict,
    )


def _breakeven_synergies(acq: Acquirer, tgt: Target, terms: DealTerms,
                         new_shares: float, atx_int: float, atx_foregone: float) -> float:
    """Pre-tax annual synergies that make the deal exactly EPS-neutral.

    Solve pf_eps == acq.eps for after-tax synergies, then gross up by the tax
    rate. Negative means the deal is already accretive with zero synergies.
    """
    pf_shares = acq.shares + new_shares
    target_pf_ni = acq.eps * pf_shares                       # NI needed for EPS parity
    base_pf_ni = acq.net_income + tgt.net_income - atx_int - atx_foregone
    needed_after_tax_syn = target_pf_ni - base_pf_ni
    return needed_after_tax_syn / (1.0 - acq.tax_rate) if acq.tax_rate < 1 else float("nan")


def sensitivity_premium_stock(acq: Acquirer, tgt: Target, terms: DealTerms,
                              premiums: list[float], stock_mixes: list[float]) -> pd.DataFrame:
    """Accretion/(dilution) % across a premium × %-stock grid.

    At each %-stock the remaining consideration keeps the original cash:debt
    ratio. Rows = premium, columns = % stock.
    """
    base_cash_debt = terms.pct_cash + terms.pct_debt
    out = np.zeros((len(premiums), len(stock_mixes)))
    for i, prem in enumerate(premiums):
        for j, stock in enumerate(stock_mixes):
            rest = 1.0 - stock
            if base_cash_debt > 0:
                pct_cash = rest * (terms.pct_cash / base_cash_debt)
                pct_debt = rest * (terms.pct_debt / base_cash_debt)
            else:
                pct_cash, pct_debt = 0.0, rest
            t = DealTerms(
                offer_premium=prem, pct_stock=stock, pct_cash=pct_cash, pct_debt=pct_debt,
                new_debt_rate=terms.new_debt_rate, cash_yield=terms.cash_yield,
                pretax_synergies=terms.pretax_synergies,
                assume_target_debt=terms.assume_target_debt,
                transaction_fees=terms.transaction_fees,
            )
            out[i, j] = run_deal(acq, tgt, t).accretion_dilution * 100.0
    return pd.DataFrame(out, index=[f"{p:.0%}" for p in premiums],
                        columns=[f"{s:.0%}" for s in stock_mixes])


def sensitivity_synergies_premium(acq: Acquirer, tgt: Target, terms: DealTerms,
                                  synergies: list[float], premiums: list[float]) -> pd.DataFrame:
    """Accretion/(dilution) % across a synergies × premium grid.

    Rows = pre-tax synergies ($), columns = premium.
    """
    out = np.zeros((len(synergies), len(premiums)))
    for i, syn in enumerate(synergies):
        for j, prem in enumerate(premiums):
            t = DealTerms(
                offer_premium=prem, pct_stock=terms.pct_stock, pct_cash=terms.pct_cash,
                pct_debt=terms.pct_debt, new_debt_rate=terms.new_debt_rate,
                cash_yield=terms.cash_yield, pretax_synergies=syn,
                assume_target_debt=terms.assume_target_debt,
                transaction_fees=terms.transaction_fees,
            )
            out[i, j] = run_deal(acq, tgt, t).accretion_dilution * 100.0
    return pd.DataFrame(out, index=[f"{s:,.0f}" for s in synergies],
                        columns=[f"{p:.0%}" for p in premiums])
