"""
Factor Validation Engine — blueprint #4.

"This is where backtests lie." Built to five institutional constraints so the t-stats and
decay curves are real, not phantom alpha from naive OLS on overlapping returns:

  1. NEWEY-WEST overlapping-return correction — Fama-MacBeth gamma_t series from a multi-day
     holding period tau is MA(tau-1) autocorrelated. The t-stat on mean gamma uses a
     Newey-West (Bartlett kernel) long-run variance with lag L = tau - 1 (>=3 when tau=1).
  2. WLS, not OLS — cross-sectional regressions weight each name by sqrt(ADTV_20) so an
     illiquid micro-cap printing +15% on no volume can't warp the factor return gamma_t.
  3. SPEARMAN rank IC — signals are built to preserve cross-sectional ORDERING, not predict
     point returns; rank IC is the right, outlier-robust metric.
  4. MARGINAL decay, not cumulative — the SUE half-life H is fit to the MARGINAL IC
     (corr of Z_t with the single-day return on day t+k), so early alpha doesn't bleed
     forward and create a false illusion of freshness.
  5. CIRCUIT CENSORING — a name locked at its circuit on the formation day (O==H==L==C) has
     no ask liquidity; it is excluded from formation so we never book an un-executable fill.

Point-in-time inputs: prices from `adjusted_prices` (TRI, split/bonus clean); SUE reconstructed
through `fundamental_bridge` knowledge dates (no look-ahead). v1 validates the two cleanly
reconstructable market-wide signals — MOMENTUM and SUE — which is exactly what fits H + weights.
Value/Quality need point-in-time fundamentals (the screener snapshot problem) — deferred.

Output: data/validation/ic_decay.parquet, fama_macbeth.parquet + console report.
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "quant_signals.duckdb"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "validation"

FORMATION_STEP = 5      # weekly formation dates (trading days) -> quasi-independent samples
MAX_HORIZON = 60        # IC decay measured to t+60
MIN_NAMES = 30          # min cross-section size to run a regression / IC on a date
MIN_ADTV_LAKH = 100.0   # tradable universe: >= Rs 1cr/day avg turnover


# ───────────────────────── stats primitives ─────────────────────────
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Cross-sectional Spearman rank IC over the names valid in both."""
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < MIN_NAMES:
        return np.nan
    rx = pd.Series(x[m]).rank().to_numpy()
    ry = pd.Series(y[m]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _newey_west_tstat(g: np.ndarray, L: int):
    """t-stat of mean(g) with a Newey-West (Bartlett) long-run variance, lag L."""
    g = g[~np.isnan(g)]
    T = len(g)
    if T < max(5, L + 2):
        return np.nan, np.nan, T
    gbar = g.mean()
    d = g - gbar
    S0 = np.mean(d * d)
    lrv = S0
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1)
        cov = np.mean(d[l:] * d[:-l])
        lrv += 2.0 * w * cov
    se = np.sqrt(max(lrv, 1e-18) / T)
    return float(gbar / se), float(gbar), T


def _wls_slope(z: np.ndarray, r: np.ndarray, w: np.ndarray) -> float:
    """Weighted least-squares slope of r on z (with intercept). Weights w."""
    m = ~(np.isnan(z) | np.isnan(r) | np.isnan(w))
    if m.sum() < MIN_NAMES:
        return np.nan
    z, r, w = z[m], r[m], w[m]
    X = np.column_stack([np.ones_like(z), z])
    W = w
    XtW = X.T * W
    try:
        beta = np.linalg.solve(XtW @ X, XtW @ r)
    except np.linalg.LinAlgError:
        return np.nan
    return float(beta[1])


# ───────────────────────── panels ─────────────────────────
def _load_panels(con):
    px = con.execute("""
        SELECT trade_date, canonical_symbol AS symbol, adj_close, turnover,
               (open = high AND high = low AND low = adj_close) AS locked
        FROM adjusted_prices
        WHERE adj_close > 0
    """).fetchdf()
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["symbol", "trade_date"]).drop_duplicates(["trade_date", "symbol"])

    close = px.pivot(index="trade_date", columns="symbol", values="adj_close").sort_index()
    turn = px.pivot(index="trade_date", columns="symbol", values="turnover").reindex_like(close)
    locked = px.pivot(index="trade_date", columns="symbol", values="locked").reindex_like(close).fillna(False)
    return close, turn, locked


def _momentum_panel(close: pd.DataFrame) -> pd.DataFrame:
    """12-1 momentum at every date (skip last month)."""
    return close.shift(21) / close.shift(252) - 1.0


def _sue_panel(con, dates: pd.DatetimeIndex, symbols) -> pd.DataFrame:
    """Reconstruct point-in-time SUE: at each date, use only quarters whose knowledge_date
    has already passed (no look-ahead), 8-quarter standardization."""
    fb = con.execute("""
        SELECT symbol, effective_date, knowledge_date, eps
        FROM fundamental_bridge WHERE eps IS NOT NULL
    """).fetchdf()
    fb["knowledge_date"] = pd.to_datetime(fb["knowledge_date"])
    fb["effective_date"] = pd.to_datetime(fb["effective_date"])
    fb = fb.sort_values(["symbol", "effective_date"])

    def _sue(eps_arr):
        if len(eps_arr) < 8:
            return np.nan
        a = eps_arr[-8:][::-1]  # newest first
        if np.any(np.isnan(a)):
            return np.nan
        yoy = a[0] - a[4]
        mean_chg = (a[0] - a[7]) / 7
        dev = sum(((a[i] - a[i + 1]) - mean_chg) ** 2 for i in range(7))
        std = (dev / 7) ** 0.5
        return yoy / std if std > 0 else np.nan

    # Per symbol, SUE steps at each knowledge_date using quarters known by then.
    steps = {}  # symbol -> (sorted knowledge_dates array, sue values array)
    for sym, g in fb.groupby("symbol"):
        g = g.sort_values("knowledge_date")
        kds, vals = [], []
        eps_known = []
        for r in g.itertuples(index=False):
            eps_known.append(r.eps)               # quarter now public
            kds.append(r.knowledge_date)
            vals.append(_sue(np.array(eps_known, dtype=float)))
        steps[sym] = (np.array(kds, dtype="datetime64[ns]"), np.array(vals, dtype=float))

    out = pd.DataFrame(index=dates, columns=symbols, dtype=float)
    dvals = dates.values
    for sym in symbols:
        if sym not in steps:
            continue
        kds, vals = steps[sym]
        # asof: latest knowledge_date STRICTLY before each formation date (T+1 discipline)
        pos = np.searchsorted(kds, dvals, side="left") - 1
        col = np.where(pos >= 0, vals[np.clip(pos, 0, len(vals) - 1)], np.nan)
        out[sym] = col
    return out


# ───────────────────────── engines ─────────────────────────
def ic_decay(signal: pd.DataFrame, close: pd.DataFrame, locked: pd.DataFrame,
             adtv: pd.DataFrame, formation: list, name: str) -> pd.DataFrame:
    """Per-horizon Spearman IC suite:
      - MARGINAL  IC: corr(Z_t, single-day return on day t+k)   -> fit the decay SHAPE / half-life
      - CUMULATIVE IC: corr(Z_t, R_{t->t+k})                     -> total directional accuracy
      - IC-IR     : mean(IC_t) / std(IC_t) across formation dates -> CONSISTENCY (Grinold)
    Marginal single-day IC is microstructure-noisy and used only for the curve shape; capital
    decisions should look at cumulative IC and especially IC-IR."""
    single_fwd = close.shift(-1) / close - 1.0          # return realised over (t, t+1]
    idx = {d: i for i, d in enumerate(close.index)}

    def _stats(a):
        a = np.asarray(a, dtype=float)
        if len(a) < 2:
            return (np.nan, np.nan, np.nan)
        m, s = a.mean(), a.std()
        return (m, s, m / s if s > 0 else np.nan)   # mean, std, IC-IR

    rows = []
    for k in range(1, MAX_HORIZON + 1):
        marg, cum = [], []
        for f in formation:
            i = idx[f]
            if i + k >= len(close.index):
                continue
            z = signal.iloc[i].to_numpy(dtype=float)
            bad = locked.iloc[i].to_numpy(dtype=bool) | (adtv.iloc[i].to_numpy(dtype=float) < MIN_ADTV_LAKH)
            z = np.where(bad, np.nan, z)                       # circuit + liquidity censor
            im = _spearman(z, single_fwd.iloc[i + k].to_numpy(dtype=float))
            ic = _spearman(z, (close.iloc[i + k] / close.iloc[i] - 1.0).to_numpy(dtype=float))
            if not np.isnan(im):
                marg.append(im)
            if not np.isnan(ic):
                cum.append(ic)
        mm, cc = _stats(marg), _stats(cum)
        rows.append({"signal": name, "horizon": k,
                     "ic_marginal": mm[0], "ir_marginal": mm[2],
                     "ic_cumulative": cc[0], "ir_cumulative": cc[2],
                     "ic_std": mm[1], "n_dates": len(marg)})
    return pd.DataFrame(rows)


def fama_macbeth(signal: pd.DataFrame, close: pd.DataFrame, locked: pd.DataFrame,
                 adtv: pd.DataFrame, formation: list, tau: int, name: str) -> dict:
    """WLS cross-sectional regressions of the tau-day forward return on the signal;
    Newey-West t-stat (lag = tau-1, floor 3) on the gamma_t series."""
    idx = {d: i for i, d in enumerate(close.index)}
    gammas = []
    for f in formation:
        i = idx[f]
        if i + tau >= len(close.index):
            continue
        z = signal.iloc[i].to_numpy(dtype=float)
        bad = locked.iloc[i].to_numpy(dtype=bool) | (adtv.iloc[i].to_numpy(dtype=float) < MIN_ADTV_LAKH)
        z = np.where(bad, np.nan, z)
        # cross-sectional standardize the signal so gamma is comparable across dates
        mu, sd = np.nanmean(z), np.nanstd(z)
        z = (z - mu) / sd if sd and not np.isnan(sd) else z
        r = (close.iloc[i + tau] / close.iloc[i] - 1.0).to_numpy(dtype=float)
        w = np.sqrt(np.clip(adtv.iloc[i].to_numpy(dtype=float), 0, None))
        g = _wls_slope(z, r, w)
        if not np.isnan(g):
            gammas.append(g)
    gammas = np.array(gammas)
    L = max(tau - 1, 3)
    t_nw, gbar, T = _newey_west_tstat(gammas, L)
    # naive iid t-stat for contrast (shows the inflation)
    t_iid = gbar / (np.nanstd(gammas) / np.sqrt(T)) if T and np.nanstd(gammas) > 0 else np.nan
    return {"signal": name, "tau": tau, "nw_lag": L, "mean_gamma": gbar,
            "t_newey_west": t_nw, "t_iid_naive": t_iid, "n_periods": T}


def fit_half_life(decay: pd.DataFrame) -> float:
    """Fit IC_marg(k) ~ IC0 * 0.5^(k/H) over the positive-IC region; return H (trading days)."""
    d = decay[(decay["ic_marginal"] > 0) & (decay["horizon"] <= 40)]
    if len(d) < 5:
        return np.nan
    k = d["horizon"].to_numpy(dtype=float)
    y = np.log(d["ic_marginal"].to_numpy(dtype=float))
    # log(IC) = log(IC0) - (ln2/H) k  -> linear fit
    slope, _ = np.polyfit(k, y, 1)
    if slope >= 0:
        return np.nan
    return float(np.log(2) / (-slope))


def run(con=None):
    should_close = con is None
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    close, turn, locked = _load_panels(con)
    adtv = turn.rolling(20, min_periods=10).mean()       # ADTV_20 (lakh)
    dates = close.index

    # formation dates: weekly, only where a 252d lookback exists and 60d forward remains
    valid = dates[(np.arange(len(dates)) >= 252) & (np.arange(len(dates)) < len(dates) - MAX_HORIZON)]
    formation = list(valid[::FORMATION_STEP])
    print(f"  {len(dates)} trading days, {len(formation)} weekly formation dates "
          f"({formation[0].date()} → {formation[-1].date()})")

    mom = _momentum_panel(close)
    sue = _sue_panel(con, dates, list(close.columns))
    print(f"  panels built: momentum {int(mom.notna().sum().sum()):,} obs, "
          f"SUE {int(sue.notna().sum().sum()):,} obs")
    if should_close:
        con.close()

    # ── IC decay curves (marginal, Spearman, censored) ──
    decay = pd.concat([
        ic_decay(mom, close, locked, adtv, formation, "momentum_12_1"),
        ic_decay(sue, close, locked, adtv, formation, "sue"),
    ], ignore_index=True)
    decay.to_parquet(OUT_DIR / "ic_decay.parquet", index=False)

    # ── Fama-MacBeth WLS + Newey-West at natural horizons ──
    fm = pd.DataFrame([
        fama_macbeth(mom, close, locked, adtv, formation, tau=21, name="momentum_12_1"),
        fama_macbeth(sue, close, locked, adtv, formation, tau=21, name="sue"),
        fama_macbeth(sue, close, locked, adtv, formation, tau=5, name="sue_5d"),
    ])
    fm.to_parquet(OUT_DIR / "fama_macbeth.parquet", index=False)

    # ── report ──
    print("\n=== IC SUITE (Spearman, censored) ===")
    for nm in ["momentum_12_1", "sue"]:
        d = decay[decay.signal == nm]
        H = fit_half_life(d)
        def at(k, col):
            v = d[d.horizon == k][col]
            return v.values[0] if len(v) else np.nan
        print(f"  {nm}:")
        print(f"     marginal IC   k=1:{at(1,'ic_marginal'):+.3f}  k=5:{at(5,'ic_marginal'):+.3f}  "
              f"k=21:{at(21,'ic_marginal'):+.3f}  k=60:{at(60,'ic_marginal'):+.3f}   half-life H="
              + (f"{H:.0f} td" if not np.isnan(H) else "n/a"))
        print(f"     cumulative IC k=5:{at(5,'ic_cumulative'):+.3f}  k=21:{at(21,'ic_cumulative'):+.3f}  "
              f"k=60:{at(60,'ic_cumulative'):+.3f}")
        print(f"     IC-IR (cumul) k=5:{at(5,'ir_cumulative'):+.2f}  k=21:{at(21,'ir_cumulative'):+.2f}  "
              f"k=60:{at(60,'ir_cumulative'):+.2f}   (mean IC / std IC — consistency)")

    print("\n=== FAMA-MACBETH (WLS by sqrt(ADTV), Newey-West vs naive-iid t-stats) ===")
    print(fm.round(3).to_string(index=False))
    print(f"\n  (NW t-stat < iid t-stat = the overlap/serial-correlation inflation correctly removed)")
    print(f"\nSaved: {OUT_DIR/'ic_decay.parquet'}, {OUT_DIR/'fama_macbeth.parquet'}")
    return decay, fm


if __name__ == "__main__":
    run()
