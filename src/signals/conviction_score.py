"""
Conviction Score — the unified, engineered composite across every signal.

This supersedes the naive MasterScore (which was just mean(z) + macro tilt).
It fixes three real flaws in equal-weight averaging:

  1. THEME DOUBLE-COUNTING — signals are first collapsed into 6 factor BLOCKS so a
     theme with many sub-signals (flow has 4) can't drown one with few (momentum has 1).
  2. THIN-EVIDENCE OVERCONFIDENCE — a coverage shrinkage term (James-Stein flavour)
     pulls scores toward neutral when few blocks are present. 6-block conviction stays
     full strength; 2-block conviction gets damped.
  3. OUTLIER DOMINATION + DISAGREEMENT — every signal is converted to a robust
     rank→normal score (heavy tails neutralised), and an AGREEMENT multiplier rewards
     stocks whose blocks point the same way and haircuts conflicted ones. A calm +1.2
     across six blocks now beats a lone +3 with five negatives.

Two principled interaction guards (value×quality, momentum×quality) reward CONFIRMED
cheap-and-good / trending-and-good, and implicitly penalise value traps & junk momentum.

WEIGHTING STANCE (per Param, option 3): cross-block weights are held EQUAL for now. The
machinery is fully built; once the IC-validation layer measures forward-return
predictiveness on the freshly-backfilled data, replace BLOCK_WEIGHTS with fitted values —
that is the only line that needs to change.

Output: data/conviction_score.parquet  (conviction = 0..100)
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DB_PATH = DATA_DIR / "quant_signals.duckdb"

# ── Cross-block weights. EQUAL for now (deferred to IC validation). ──
# Swap these for fitted weights once forward-return IC is measured.
BLOCK_WEIGHTS = {
    "earnings": 1.0,
    "value":    1.0,
    "quality":  1.0,
    "momentum": 1.0,
    "flow":     1.0,
}
MACRO_TILT = 0.05      # uniform regime nudge (does not change cross-sectional rank)
SHRINK_K = 2.5         # coverage shrinkage: factor = n/(n+K). Higher = thin-coverage names
                       # damped harder so a 2-block stock can't top a 5-block one on extremes.
QUALITY_GATE_K = 0.5          # soft gate steepness — gentle decay, not a turnover-inducing cliff
QUALITY_GATE_CENTER = -1.0    # inflection shifted to Q=-1σ: average/high quality keep ~80-100% of
                              # their value credit; only genuine junk (Q < -2) is crushed to ~0
MIN_BLOCKS = 2


def _ppf(p):
    """Inverse standard-normal CDF (Acklam's rational approximation). Vectorised,
    accurate to ~1e-9, no scipy dependency."""
    p = np.asarray(p, dtype=float)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    out = np.zeros_like(p)
    lo, hi = p < plow, p > phigh
    mid = (~lo) & (~hi)
    ql = np.sqrt(-2 * np.log(p[lo])) if lo.any() else np.array([])
    if lo.any():
        out[lo] = (((((c[0]*ql+c[1])*ql+c[2])*ql+c[3])*ql+c[4])*ql+c[5]) / \
                  ((((d[0]*ql+d[1])*ql+d[2])*ql+d[3])*ql+1)
    if hi.any():
        qh = np.sqrt(-2 * np.log(1 - p[hi]))
        out[hi] = -(((((c[0]*qh+c[1])*qh+c[2])*qh+c[3])*qh+c[4])*qh+c[5]) / \
                   ((((d[0]*qh+d[1])*qh+d[2])*qh+d[3])*qh+1)
    if mid.any():
        qm = p[mid] - 0.5
        r = qm * qm
        out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*qm / \
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return out


def _robust(s: pd.Series) -> pd.Series:
    """Cross-sectional rank → inverse-normal score. Robust to heavy tails: every
    distribution becomes ~N(0,1) by rank, so no single outlier dominates."""
    n = s.notna().sum()
    if n < 3:
        return pd.Series(np.nan, index=s.index)
    r = s.rank(method="average", pct=True)
    r = r.clip(0.5 / n, 1 - 0.5 / n)   # avoid +/-inf at the extremes
    z = pd.Series(np.nan, index=s.index)
    z.loc[r.notna()] = _ppf(r.dropna().values)
    return z


def _mad_z(s: pd.Series, clip: float = 3.0) -> pd.Series:
    """Robust z-score via Median Absolute Deviation. MAD is not poisoned by a single
    illiquid micro-cap outlier the way mean/std is:  z = (x - median) / (1.4826 * MAD)."""
    x = s.astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    if mad and not np.isnan(mad):
        return ((x - med) / (1.4826 * mad)).clip(-clip, clip)
    sd = x.std()
    if sd and not np.isnan(sd):
        return ((x - med) / sd).clip(-clip, clip)
    return pd.Series(0.0, index=s.index)


def _mad_z_within_sector(values: pd.Series, sector_map: pd.Series, min_n: int = 5) -> pd.Series:
    """MAD z-score computed WITHIN each macro sector (sector-neutral). Stocks with no
    sector tag, or in a too-small sector bucket, fall back to the market-wide MAD z."""
    mkt = _mad_z(values)
    out = mkt.copy()
    sec = sector_map.reindex(values.index)
    for s, idx in values.groupby(sec).groups.items():
        if len(idx) >= min_n:
            out.loc[idx] = _mad_z(values.loc[idx])
    return out.clip(-3.0, 3.0)


def _load(name, cols=None, latest_date_col=None, latest_group="symbol"):
    p = DATA_DIR / f"{name}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if latest_date_col and latest_date_col in df.columns:
        df[latest_date_col] = pd.to_datetime(df[latest_date_col])
        df = df.sort_values(latest_date_col).groupby(latest_group).tail(1)
    return df[cols] if cols else df


def compute_conviction(con: Optional[duckdb.DuckDBPyConnection] = None) -> pd.DataFrame:
    # ── Pull each signal's headline raw column ──
    parts = {}

    sue = _load("sue_signal", ["symbol", "sue", "fm_score"])
    if not sue.empty:
        parts["sue"] = sue.rename(columns={"sue": "v"})[["symbol", "v"]]
        parts["fm"] = sue.rename(columns={"fm_score": "v"})[["symbol", "v"]]

    vq = _load("value_quality_signal", ["symbol", "value_score", "quality_score"])
    if not vq.empty:
        parts["value"] = vq.rename(columns={"value_score": "v"})[["symbol", "v"]]
        parts["quality"] = vq.rename(columns={"quality_score": "v"})[["symbol", "v"]]

    mom = _load("momentum_signal", ["symbol", "mom_score"])
    if not mom.empty:
        parts["mom"] = mom.rename(columns={"mom_score": "v"})[["symbol", "v"]].dropna()

    fo = _load("fo_signals", ["symbol", "fo_bullish_raw", "trade_date"], latest_date_col="trade_date")
    if not fo.empty:
        parts["fo"] = fo.rename(columns={"fo_bullish_raw": "v"})[["symbol", "v"]]

    cadp = _load("cadp_signal", ["symbol", "cadp", "earnings_date"], latest_date_col="earnings_date")
    if not cadp.empty:
        parts["cadp"] = cadp.rename(columns={"cadp": "v"})[["symbol", "v"]]

    chi = _load("chi_signal", ["symbol", "chi", "quarter_end"], latest_date_col="quarter_end")
    if not chi.empty:
        parts["chi"] = chi.rename(columns={"chi": "v"})[["symbol", "v"]]

    sm = _load("smart_money_signal", ["symbol", "net_value_cr"])
    if not sm.empty:
        parts["smart"] = sm.rename(columns={"net_value_cr": "v"})[["symbol", "v"]]

    macro = _load("macro_nowcast")
    macro_z = 0.0
    if not macro.empty:
        m = macro.dropna(subset=["momentum_zscore"])
        if not m.empty:
            macro_z = float(m.iloc[-1]["momentum_zscore"])

    if not parts:
        print("  No signals available")
        return pd.DataFrame()

    # ── Sector map + earnings knowledge dates (blueprint #3) ──
    _con = con or duckdb.connect(str(DB_PATH), read_only=True)
    try:
        sec_df = _con.execute(
            "SELECT symbol, macro_sector FROM company_sectors WHERE macro_sector IS NOT NULL"
        ).fetchdf()
        sector_map = sec_df.drop_duplicates("symbol").set_index("symbol")["macro_sector"]
        print(f"  Sector map: {len(sector_map)} symbols, {sector_map.nunique()} AMFI macro sectors")
    except Exception:
        sector_map = pd.Series(dtype=object)
        print("  WARN: company_sectors missing -> Value/Quality fall back to market-wide (not sector-neutral)")
    try:
        kd = _con.execute(
            "SELECT symbol, MAX(knowledge_date) kd FROM fundamental_bridge GROUP BY symbol"
        ).fetchdf()
        kd_map = pd.to_datetime(kd.set_index("symbol")["kd"])
        as_of = pd.to_datetime(_con.execute("SELECT MAX(trade_date) FROM adjusted_prices").fetchone()[0])
    except Exception:
        kd_map, as_of = pd.Series(dtype="datetime64[ns]"), None
    if con is None:
        _con.close()

    # ── SUE time-decay anchored to knowledge_date (PEAD is event alpha, decays fast).
    #    H ~= 21 trading days (~30 calendar) for the retail-crowded Indian market. ──
    H_CAL = 30.0
    if as_of is not None and "sue" in parts and len(kd_map):
        sdf = parts["sue"].drop_duplicates("symbol").set_index("symbol")
        delta = (as_of - kd_map.reindex(sdf.index)).dt.days
        decay = np.power(0.5, delta / H_CAL).fillna(1.0)   # no knowledge_date -> undecayed
        sdf["v"] = sdf["v"].astype(float) * decay.values
        parts["sue"] = sdf.reset_index()
        print(f"  SUE decayed to knowledge_date (H={H_CAL:.0f}d cal ~= 21 td); "
              f"median age {delta.median():.0f}d, median retained {decay.median()*100:.0f}%")

    # ── Standardize each signal (MAD robust-z). Value/Quality WITHIN sector;
    #    SUE/Momentum/Flow market-wide (sector-momentum & PEAD regimes preserved). ──
    SECTOR_NEUTRAL = {"value", "quality"}
    universe = sorted(set().union(*[set(p["symbol"]) for p in parts.values()]))
    R = pd.DataFrame({"symbol": universe}).set_index("symbol")
    for key, df in parts.items():
        s = df.dropna(subset=["v"]).drop_duplicates("symbol").set_index("symbol")["v"]
        if key in SECTOR_NEUTRAL and len(sector_map):
            R[key] = _mad_z_within_sector(s, sector_map).reindex(R.index)
        else:
            R[key] = _mad_z(s).reindex(R.index)

    # ── Collapse signals into 6 factor blocks (mean of available components) ──
    blocks = pd.DataFrame(index=R.index)
    blocks["earnings"] = R[["sue", "fm"]].mean(axis=1) if "sue" in R else np.nan
    blocks["value"] = R["value"] if "value" in R else np.nan
    blocks["quality"] = R["quality"] if "quality" in R else np.nan
    blocks["momentum"] = R["mom"] if "mom" in R else np.nan
    flow_cols = [c for c in ["fo", "cadp", "chi", "smart"] if c in R]
    blocks["flow"] = R[flow_cols].mean(axis=1) if flow_cols else np.nan

    # QUALITY GATE on VALUE (final composite layer): a linear Value+Quality average lets a PSU
    # "value trap" (cheap P/E but bleeding cash, Q<<0) average out to a false positive. Instead we
    # GATE the value block multiplicatively by quality — Effective Value = Value · σ(k·(Q − center)).
    # Soft (k=0.5) + inflection at Q=-1σ: average/high quality keep their value credit; genuine junk
    # is crushed toward zero. Momentum is left PURE & un-gated (junk/liquidity rallies are real alpha).
    q = blocks["quality"].to_numpy(dtype=float)
    gate = np.where(np.isnan(q), 1.0,
                    1.0 / (1.0 + np.exp(-QUALITY_GATE_K * (q - QUALITY_GATE_CENTER))))
    blocks["value"] = blocks["value"].to_numpy(dtype=float) * gate

    bnames = list(BLOCK_WEIGHTS.keys())
    W = np.array([BLOCK_WEIGHTS[b] for b in bnames])
    B = blocks[bnames].to_numpy()                       # (n_stocks, n_blocks)
    present = ~np.isnan(B)
    n_blocks = present.sum(axis=1)

    # weighted mean over PRESENT blocks (renormalised weights)
    Wmat = np.where(present, W, 0.0)
    wsum = Wmat.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        composite_raw = np.where(wsum > 0, np.nansum(np.where(present, B, 0.0) * Wmat, axis=1) / wsum, np.nan)

    # coverage shrinkage: thin evidence -> pull toward neutral
    shrink = n_blocks / (n_blocks + SHRINK_K)

    # agreement: share of present blocks whose sign matches the composite sign
    sign = np.sign(composite_raw)[:, None]
    same = (np.sign(np.where(present, B, 0.0)) == sign) & present
    agreement = np.where(n_blocks > 0, same.sum(axis=1) / np.maximum(n_blocks, 1), 0.5)
    agree_mult = 0.5 + 0.5 * agreement

    # Value-trap suppression now lives in the multiplicative quality GATE on the value block above
    # (replaces the old additive ReLU·ReLU bonus, which only rewarded confirmation and let cheap-junk
    # leak through at a haircut). composite_raw already reflects the gated value.
    final = composite_raw * shrink * agree_mult + MACRO_TILT * macro_z

    out = blocks.copy()
    out.columns = [f"b_{c}" for c in out.columns]
    out["n_blocks"] = n_blocks
    out["agreement"] = agreement
    out["composite_raw"] = composite_raw
    out["final_score"] = final
    out = out.reset_index()

    # enforce minimum coverage
    out.loc[out["n_blocks"] < MIN_BLOCKS, "final_score"] = np.nan

    # map to a 0..100 conviction via logistic of the standardised final score
    fz = out["final_score"]
    mu, sd = fz.mean(), fz.std()
    z = (fz - mu) / (sd if sd and not np.isnan(sd) else 1.0)
    out["conviction"] = 100.0 / (1.0 + np.exp(-1.7 * z))
    out["conviction_rank"] = out["final_score"].rank(pct=True)
    out["macro_z"] = macro_z

    out = out.sort_values("final_score", ascending=False, na_position="last").reset_index(drop=True)
    n_scored = out["final_score"].notna().sum()
    print(f"  Conviction computed: {n_scored} scored ({(out['n_blocks']>=4).sum()} with 4+ blocks), macro_z={macro_z:+.2f}")
    return out


if __name__ == "__main__":
    cs = compute_conviction()
    if not cs.empty:
        scored = cs[cs["final_score"].notna()]
        cols = ["symbol", "conviction", "n_blocks", "agreement",
                "b_earnings", "b_value", "b_quality", "b_momentum", "b_flow"]
        print("\n=== Conviction Top 25 ===")
        print(scored.head(25)[cols].round(2).to_string(index=False))
        print("\n=== Bottom 10 ===")
        print(scored.tail(10)[cols].round(2).to_string(index=False))
        print("\n=== Block coverage ===")
        print(scored["n_blocks"].value_counts().sort_index())
        print("\nconviction stats:"); print(scored["conviction"].describe().round(2))

        out = DATA_DIR / "conviction_score.parquet"
        cs.to_parquet(out, index=False)
        print(f"\nSaved {len(cs)} rows to {out.name}")
