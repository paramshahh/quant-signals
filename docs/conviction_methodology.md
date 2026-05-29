# Conviction Score — Methodology

The **Conviction Score** is the unified composite that fuses every signal into one 0–100
number. It supersedes the naive MasterScore (`mean(z) + 0.1·macro`), which had three flaws:
theme double-counting, thin-evidence overconfidence, and outlier domination.

Implementation: `src/signals/conviction_score.py`. Output: `data/conviction_score.parquet`.

---

## Construction, step by step (and the literature behind each choice)

| # | Step | What it does | Why — literature |
|---|------|--------------|------------------|
| 1 | **MAD robust-z, sector-aware** | Each signal → robust z `(x−median)/(1.4826·MAD)`, clip ±3σ (MAD not poisoned by micro-cap outliers). **Value & Quality standardized WITHIN AMFI macro sector** (sector-neutral); **SUE, Momentum, Flow market-wide**. | Robust standardization (MAD) over mean/std; sector-neutral value/quality avoids the PSU/oil value-trap. |
| 2 | **6 factor blocks** | Collapse correlated signals into themes: `Earnings` (SUE+FM), `Value`, `Quality`, `Momentum`, `Flow` (F&O+CADP+CHI+smart-money), + a `Macro` tilt. Average *within* a block, weight *across* blocks — so a theme with many sub-signals can't cast extra votes. | Factor-zoo redundancy: **Cochrane (2011)**, **Harvey, Liu & Zhu (2016)**, **Green, Hand & Zhang (2017)**. |
| 3 | **Coverage shrinkage** | Composite × `n_blocks / (n_blocks + 2.5)`. Thin-evidence names pulled toward neutral; full-coverage names keep full strength. | Shrinkage estimators dominate MLE: **James & Stein (1961)**; **Jorion (1986)**; **Ledoit & Wolf (2004)**. |
| 4 | **Agreement multiplier** | × `(0.5 + 0.5·agreement)`, where agreement = share of present blocks whose sign matches the composite. Consensus rewarded, conflict haircut. | Averaging independent signals raises effective IC: Fundamental Law of Active Management — **Grinold (1989)**. |
| 5 | **Quality gate on Value** | `Effective Value = Value · σ(k·(Quality − center))`, soft k=0.5, inflection shifted to Quality=−1σ. Cheap-but-junk (PSU value trap: low P/E, Q≪0) has its value credit crushed toward 0; average/high quality keep ~80–100%. Multiplicative GATE (not the old additive ReLU bonus, which only haircut traps). **Momentum is left PURE / un-gated** — junk/liquidity rallies are real Indian-equity alpha. | Value traps: **Piotroski (2000)**. Quality: **Novy-Marx (2013)**, **Asness, Frazzini & Pedersen (2019) "Quality Minus Junk"**. Soft gate avoids turnover-inducing cliffs. |
| 6 | **0–100 map** | Standardize the final score, map through a logistic (`100/(1+e^(−1.7z))`). 50 = median. | Monotone squashing for interpretability. |

Momentum block itself rests on **Jegadeesh & Titman (1993)** and the 52-week-high effect of **George & Hwang (2004)**.

### Sector source & SUE decay (added in blueprint #3)
- **Sector neutralization** uses the canonical **AMFI Macro-Economic Sector** (the `IN0X` codes, e.g.
  IN05 Financial Services, IN08 IT) captured natively from screener.in's `/market/IN0X/` hierarchy —
  not the dirty NSE-filings `industry` tag. Static guardrail = the canonical IN-code set (flags UNMAPPED drift).
- **SUE time-decay**: SUE is multiplied by `0.5^(Δ/H)`, Δ = days since the PEAD became public
  (the PIT `knowledge_date`), `H ≈ 21 trading days (~30 cal)` — the retail-crowded Indian market
  corrects the underreaction faster than the classic US ~50-day Bernard-Thomas figure. A 15-day-old
  beat retains ~71%; stale beats fade out of the composite. (`H` is a prior pending IC calibration in #4.)

---

## The weighting stance (important)

Cross-block weights are **EQUAL** (`BLOCK_WEIGHTS = {1,1,1,1,1}`), and deliberately so:

- **DeMiguel, Garlappi & Uppal (2009)** — across 14 datasets, naive 1/N beats mean-variance
  optimization out-of-sample, because estimation error swamps the optimizer's theoretical gains.
  When you can't reliably estimate weights, equal-weighting is the safer bet.
- **McLean & Pontiff (2016)** and **Harvey, Liu & Zhu (2016)** — published anomalies decay
  post-publication and most "discovered" factors are multiple-testing artifacts. Aggressively
  fitting weights now would overfit a single snapshot.

### Weights are NOT fitted to data
None of the dials — block weights, the 0.6/0.4 momentum blend, shrinkage K=2.5, λ=0.06,
winsor ±3, logistic ×1.7 — were estimated from returns. They are **priors and conventions**.
Recomputing on fresh data changes the *inputs*, never these coefficients.

The one exception, disclosed: K was nudged 1.5 → 2.5 after observing 2-block names topping the
in-sample cross-section. That is eyeballing, not validation.

### How weights become legitimate
Only the **IC-validation layer** (pending) can set evidence-based weights:
1. reconstruct each signal's values at past month-ends,
2. measure the information coefficient vs. subsequent 1/3/6-month returns,
3. set block weights from the measured ICs.

This is now *computable* (the price backfill provides forward returns) but **not yet done**.
Until then: treat the leaderboard as *structurally sensible*, not *empirically optimal*.

---

## Key references

- Asness, Frazzini & Pedersen (2019). *Quality Minus Junk.* Review of Accounting Studies.
- Asness, Moskowitz & Pedersen (2013). *Value and Momentum Everywhere.* Journal of Finance.
- Blom (1958). *Statistical Estimates and Transformed Beta-Variables.*
- Cochrane (2011). *Presidential Address: Discount Rates.* Journal of Finance.
- DeMiguel, Garlappi & Uppal (2009). *Optimal Versus Naive Diversification.* RFS.
- George & Hwang (2004). *The 52-Week High and Momentum Investing.* Journal of Finance.
- Green, Hand & Zhang (2017). *The Characteristics that Provide Independent Information…* RFS.
- Grinold (1989). *The Fundamental Law of Active Management.* JPM.
- Harvey, Liu & Zhu (2016). *…and the Cross-Section of Expected Returns.* RFS.
- James & Stein (1961). *Estimation with Quadratic Loss.* Berkeley Symposium.
- Jegadeesh & Titman (1993). *Returns to Buying Winners and Selling Losers.* Journal of Finance.
- Jorion (1986). *Bayes-Stein Estimation for Portfolio Analysis.* JFQA.
- Ledoit & Wolf (2004). *Honey, I Shrunk the Sample Covariance Matrix.* JPM.
- McLean & Pontiff (2016). *Does Academic Research Destroy Stock Return Predictability?* JF.
- Novy-Marx (2013). *The Other Side of Value: The Gross Profitability Premium.* JFE.
- Piotroski (2000). *Value Investing: …Separate Winners from Losers.* Journal of Accounting Research.
