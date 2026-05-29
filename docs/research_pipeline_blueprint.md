# Research Pipeline Blueprint

External quant review (2026-05-27) → the engineering spec to take the platform from
"computes signals" to "honest, un-leaked IC measurement." Build in dependency order.

Progress:
- [x] **#1 Corporate-Action engine / Total Return Index** — DONE 2026-05-27.
      `src/processing/total_return_index.py` → `data/adjusted_prices.parquet` + DuckDB
      table `adjusted_prices`. Backward CAF for splits (FV ratio), equity bonuses
      (B/(A+B), NCRPS excluded), dividends (1−D/cum_close). Audited vs the 5 institutional traps:
        - Trap 1 (backward/anchor): PASS — latest date all caf=1, today raw.
        - Trap 2 (cum-close dividend denominator): PASS — verified to the rupee (HCLTECH).
        - Trap 3 (same-day collision → one unified factor): PASS — 360ONE bonus+split = 0.25.
        - Trap 4 (symbol drift): FIXED — series grouped by ISIN, not ticker; DVL↔DTIL bridged
          into one continuous canonical=DTIL series. Momentum now keys off canonical_symbol.
        - Trap 5 (demergers): PARTIAL — implied pre-open factor (ex_open/prev_close, guarded
          0.30–1.02). Works when PARENT files (VEDL −65%→−6%). FAILS when filed under the spun
          entity (TATAMOTORS demerger filed under TMPV → parent unadjusted, still phantom −40%).
      Live impact: corrected 68/71 corporate-action names (MCX 12-1 −58%→+112%, conviction 61→98).
- [ ] #1b Demerger CoA ingestion (the proper Trap-5 fix): ingest NSE published Cost-of-Acquisition
      apportionment ratios + a parent↔child map, so parent history adjusts even when the action is
      filed under the resulting entity. Data-acquisition task.
- [ ] #2 Bitemporal PIT bridge   - [ ] #3 Sector neutralization
- [ ] #4 Validation engine       - [ ] #5 Execution friction

> Premise from the reviewer: we are well-positioned because we ingest **raw unadjusted
> NSE bhavcopy** (not pre-adjusted Yahoo data) and have **board-meeting timestamps** — the
> two things most retail setups lack, which makes correct PIT construction *possible*.

---

## 1. Bitemporal bridge (kills look-ahead bias)
`board_meetings` is the golden ticket. Build a `fundamental_bridge` (materialized view /
parquet) mapping each quarterly result's `effective_date` (e.g. Mar 31) →
`knowledge_date` (the board-meeting date where results were adopted).

Then daily signal state via DuckDB native `ASOF JOIN`:
```sql
SELECT p.date, p.symbol, f.pe_ratio
FROM daily_prices p
ASOF JOIN fundamental_bridge f
  ON p.symbol = f.symbol
  AND f.knowledge_date <= p.date
```
Guarantees that on May 10 the system only "knows" Mar-31 numbers if the board meeting
was on or before May 10.

## 2. Corporate-Action (CA) engine → adjusted_prices.parquet
Current momentum is contaminated: a 1:1 bonus halves raw price and reads as a −50% crash.
We have raw bhavcopy + ex-dates, so build a daily **Cumulative Adjustment Factor (CAF)**,
working backward from today T:
```
P_adj,t = P_raw,t × Π_{i=t+1..T} CAF_i
```
- **Splits & bonuses (mandatory):** 1:1 bonus → CAF = 0.5 for all dates before ex-date.
- **Dividends:** ordinary cash divs can be ignored for *price* momentum; **special divs**
  (>5% of mkt cap) need a total-return multiplier: `CAF = 1 − (Dividend / Close_cum)`.

Standalone Python module → `adjusted_prices.parquet`. Run 200DMA & 12-1M momentum
**exclusively** off this adjusted series.

## 3. Sector neutralization (kills the PSU/oil value trap)
Cross-sectionally demean (z-score) each factor **within sector**. BUT 75 granular industry
tags over ~1,500 names ≈ 20/sector → noisy z-scores, garbage if illiquid microcaps dominate.

**Fix:** map the 75 granular industries → ~11–15 broad macro-sectors (Financials, Energy,
IT, FMCG, Capital Goods, …). Z-score within these broader buckets (N > 75). Strips macro
beta while keeping robust peer comparison. Per the James-Stein refinement: shrink toward
the **sector** mean, not the market mean.

## 4. Execution friction
- **Circuit filter:** `O==H==L==C` is the clean proxy for a locked upper/lower circuit —
  forbid fills on those days (no counterparty exists).
- **√-impact cost model** in a mock portfolio layer once ICs are computed:
```
Cost = 0.0015 + σ × sqrt(S / ADV)
```
  `0.0015` ≈ 15 bps (Indian STT + exchange turnover). S = order size, ADV = avg daily
  volume, σ = daily vol. If S > ~10% of ADV the sqrt term destroys the trade's alpha —
  which correctly pushes IC-derived weights toward liquid names.

## 5. Validation engine (the "IC layer", done properly)
Beyond Spearman rank IC (exploratory only):
- **Fama-MacBeth** cross-sectional regressions of forward returns on each signal alongside
  standard risk factors (market beta, size, value, momentum). Report **Newey-West-adjusted
  t-stats**. If alpha vanishes when controlling for known factors, it's disguised beta.
- **Signal decay curves:** IC at t+1, t+2 … t+20; plot the curve, read the half-life
  (determines whether the signal survives execution latency).

---

## Build order (dependencies)
1. **CA engine / Total Return Index** — fixes the live momentum bug; prerequisite for all returns.
2. **Bitemporal PIT bridge** (`knowledge_date` + ASOF JOIN).
3. **Sector neutralization** (75 → ~12 buckets; demean within sector). *Improves the live score.*
4. **Validation engine** (Fama-MacBeth + Newey-West + decay curves).
5. **Execution layer** (circuit filter + √-impact).

Items 1 & 3 also improve the *live* Conviction Score; 2/4/5 are the backtest.

Only after 1–4 can the EQUAL block weights in `conviction_score.py` be replaced with
evidence-based ones without overfitting.
