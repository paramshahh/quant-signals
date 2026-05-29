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
- [x] **#2 Bitemporal PIT bridge** — DONE 2026-05-27. `src/processing/pit_bridge.py` →
      DuckDB table `fundamental_bridge` (13,946 rows; 89% knowledge_date via real board
      meetings, 11% via SEBI fallback). Each row bitemporal (effective_date = quarter end;
      knowledge_date = board meeting that adopted results). Guardrails:
        1. T+1 strict — serving join is `trade_date > knowledge_date` (NOT >=). Proven on
           MUTHOOTFIN: on the 2026-05-14 meeting day the backtester still sees Dec-2025
           (EPS 69.84), only seeing Mar-2026 (83.43) from 05-16.
        2. 45/60-day SEBI fallback when board date missing (60d for March/annual) — no quarter
           orphaned from coverage shrinkage.
        3. INSERT-only — restatements append a new (effective_date, knowledge_date) vintage;
           ASOF serves the version known as of each trade_date. Re-runs idempotent (+0 rows).
      Serving contract: ASOF JOIN daily prices on `p.trade_date > b.knowledge_date`.
- [x] **#3 Sector/Beta neutralization** — DONE 2026-05-27.
        - Sector source: screener.in exposes the SEBI/AMFI macro hierarchy via /market/IN0X/ links.
          Extended the scraper to capture macro_code + macro_sector + basic_industry → DuckDB
          table `company_sectors` (12 AMFI sectors IN01–IN12, ~75% of scored universe, was 42%
          on dirty filings). Static guardrail = the canonical IN-code set (no UNMAPPED drift).
        - Method: MAD robust-z `(x−median)/(1.4826·MAD)` everywhere (replaces rank-normal),
          clip ±3 (only ~5% clipped). Value & Quality standardized WITHIN macro sector
          (min 5/bucket else market-wide fallback); SUE/Momentum/Flow market-wide.
        - SUE time-decay anchored to PIT knowledge_date, H=30 cal days (~21 trading) — Indian
          retail crowding corrects PEAD faster than the US 50–60d. 15-day-old beat retains 71%.
        - Block weights EQUAL (horizon weighting deferred to #4 to avoid polluting the baseline).
        - Result: value-block top went from PSU-bank/oil concentrated to spread across 8 sectors.
- [x] **#4 Validation engine** — DONE 2026-05-27 (v1). `src/validation/factor_validation.py`
      → data/validation/{ic_decay,fama_macbeth}.parquet. All 5 constraints implemented:
      Newey-West (Bartlett, lag=tau-1) on Fama-MacBeth gamma_t; WLS by sqrt(ADTV_20); Spearman
      rank IC; MARGINAL decay (single-day fwd at t+k) for half-life fitting; circuit (O=H=L=C)
      + liquidity censoring at formation. Point-in-time: prices from TRI, SUE via PIT knowledge_date.
      FIRST-PASS RESULTS (113 weekly formations, ~2.3 yrs — small sample):
        - Newey-West works: momentum 21d t 2.73(iid)→1.38(NW); SUE 21d 4.04→2.44; SUE 5d 2.89→2.71.
        - Momentum 12-1 NOT significant post-NW in-sample (t=1.38). SUE survives (t≈2.4–2.7).
        - SUE marginal-IC decay does NOT fit a fast exponential → the H≈21 prior used in #3 is
          UNVALIDATED by this data. Do not trust the SUE decay tuning yet; widen sample / revisit.
        - Value/Quality validation deferred (needs point-in-time fundamentals; screener is a snapshot).
- [~] #4b Deep multi-regime panel (the validation needs >2.3yr / 1 bull regime). Architecture =
      one-time STATIC seed of history, daily cron left untouched. `src/ingestion/legacy_bhavcopy_backfill.py`
      built + proven: legacy CM bhavcopy gives OHLCV + turnover + ISIN back to ~2008 (no delivery,
      not needed for momentum/SUE). Plan: seed equity ~2015–2022 (covers 2018 midcap crash + 2020
      COVID) → bridges to live 2023+. Corporate actions pre-2023 via yfinance splits/dividends (TRI)
      + SEBI 45/60-day fallback for SUE knowledge_date (reviewer-blessed). MUST seed CA with prices,
      else deep history re-introduces phantom split crashes. WRDS=paid (skip), BSE scrape=ethos upgrade.
- [ ] #5 Execution friction (√-impact cost + circuit filter in a mock portfolio layer)

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
