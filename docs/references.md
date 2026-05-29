# References — academic defense for each architectural choice

Every non-obvious decision in this pipeline traces to published literature. This maps each
engineering choice to the paper that defends it — so the architecture is grounded in academic
consensus, not retail intuition. (Implementation lives in `src/`; design in
`research_pipeline_blueprint.md` and `conviction_methodology.md`.)

## 1. Statistical discipline — why no ML weight-optimizer, why Newey-West, why equal weights
- **Harvey, Liu & Zhu (2016), "…and the Cross-Section of Expected Returns" (RFS).** The "factor zoo":
  most published anomalies are false discoveries from multiple testing. → *Defends* our refusal to
  data-mine block weights and our insistence on out-of-sample multi-regime validation before trusting
  any signal.
- **López de Prado (2018), *Advances in Financial Machine Learning*.** Backtest overfitting, overlapping
  return horizons, why naive time-series CV fails; Deflated Sharpe. → *Defends* the Newey-West / Bartlett
  kernel correction (lag = τ−1) in `factor_validation.py` for our overlapping forward returns.
- **DeMiguel, Garlappi & Uppal (2009), "Optimal Versus Naive Diversification" (RFS).** Estimation error
  in the covariance matrix usually destroys Markowitz's theoretical gains; 1/N wins out-of-sample.
  → *Defends* equal block weights in `conviction_score.py` until IC validation justifies otherwise.

## 2. The factor interactions — why gate Value by Quality, leave Momentum pure
- **Piotroski (2000), "Value Investing… Winners from Losers" (JAR).** High book-to-market only pays if
  you filter out deteriorating profitability/leverage. → *Defends* the shifted-sigmoid **quality gate on
  Value** (`Value · σ(0.5·(Quality+1))`) — cheap-but-junk is suppressed.
- **Novy-Marx (2013), "The Other Side of Value: The Gross Profitability Premium" (JFE).** Profitability
  (quality) and value are negatively correlated and hedge each other. → *Defends* keeping Quality as its
  own block *and* as the gate on Value.
- **Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere" (JF).** Value & momentum are
  negatively correlated across regimes; combining them smooths the composite. → *Defends* keeping
  Momentum **un-gated and market-wide** (it's the hedge to Value), and explains our regime table
  (2022 value rotation vs 2020 momentum V-bottom).

## 3. The earnings anomaly (SUE / PEAD)
- **Bernard & Thomas (1989), "Post-Earnings-Announcement Drift" (J. Accounting & Economics).** The market
  systematically underreacts to earnings surprises. → *Defends* SUE as a **market-wide** signal (analysts
  are slow to re-rate whole sectors in a macro shift) and the PEAD-decay framing of the SUE half-life.

## 4. Indian-market specificity
- **Agarwalla, Jacob & Varma (2014), "Four-Factor Model in the Indian Equity Market" (IIM-A).** Fama-French
  size/value exist in India, but **momentum is exceptionally strong and persistent**. → *Defends* the
  weight we put on a clean (TRI-adjusted) momentum signal and the local relevance of these factors.

## Foundational (signal definitions)
- **Jegadeesh & Titman (1993)** — momentum (12-1). **George & Hwang (2004)** — 52-week-high momentum.
- **Ball & Brown (1968)** — earnings information content (SUE lineage).
- **Blom (1958)** — normal-scores / rank standardization. **James & Stein (1961)**, **Jorion (1986)**,
  **Ledoit & Wolf (2004)** — shrinkage (our coverage-shrinkage term).
- **Grinold (1989)** — Fundamental Law of Active Management (IC-IR, breadth → the agreement multiplier).
- **McLean & Pontiff (2016)** — anomalies decay post-publication (why we stay humble on weights).

---
*Prep note: know each paper's core thesis (abstract + intro + conclusion) so the engineering choices
can be attributed to the academic consensus — not read cover to cover.*
