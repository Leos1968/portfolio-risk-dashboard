# Investment Memos — a short pitch book

Three ideas, written to prove judgment rather than tooling: a **long**, a **short/avoid**,
and a **private-equity buyout**. Each is a one-to-two-page memo in the format an equity-research
or PE interviewer expects — thesis up top, valuation with a target, catalysts, and an honest
account of what would break it.

| # | Idea | Ticker | Stance | As of | Base case |
|---|------|--------|--------|-------|-----------|
| 1 | [Visa — the cheaper half of the payments duopoly](2026-07_visa_long.md) | V | **Long / Buy** | 2026-07-17 | ~$420 target, **+17%** (+18% w/ dividend) |
| 2 | [Tesla — priced for an autonomy future it hasn't earned](2026-07_tesla_short.md) | TSLA | **Avoid / hedged short** | 2026-07-17 | ~$215 target, **−44%** on a de-rate |
| 3 | [US Foods — a great LBO business at the wrong price](2026-07_us-foods_lbo.md) | USFD | **Watch-list buyout** | 2026-07-17 | ~10–12% IRR at spot; 20%+ only on a pullback |

### How these were built
Every load-bearing number was pulled from primary sources — company earnings releases, SEC
10-K/10-Q/8-K filings, and IR decks — and then independently fact-checked against a second source.
Where the check turned up an error, the memo carries the corrected figure (e.g. Visa's FY25 free
cash flow is ~$21.6B, not $21.2B; Tesla's *full-year* 2025 gross margin was 18.0%, not the 20.1%
that was actually the Q4 print). Prices and multiples are as of the stated date and move daily.

### They pair with the modeling suite
The [`modeling/`](../modeling) app in this repo is the quantitative other half: the **LBO / Returns**
tab reproduces the US Foods buyout math below, the **M&A** tab runs accretion/dilution, and the
**three-statement** model is the engine a DCF would sit on top of.

---
*Educational analysis by Jeriel De Leon — **not investment advice**, not a recommendation to buy or
sell any security, and not the view of any employer. Positions, prices, and estimates are as of the
dates shown and will change.*
