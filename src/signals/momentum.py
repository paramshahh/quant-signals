"""
Signal 7: Price Momentum

The single most-validated cross-sectional factor (Jegadeesh-Titman 1993,
Novy-Marx 2012). Built entirely from equity_daily OHLCV — the table that
previously fed only the charts.

Best-practice design choices:
  - 12-1 momentum: cumulative return from t-252 to t-21. We SKIP the most
    recent ~1 month because short-horizon returns mean-revert (the 1-month
    reversal effect would otherwise contaminate a medium-term momentum signal).
  - 6-1 momentum: faster variant (t-126 to t-21) for corroboration.
  - 52-week-high proximity (George & Hwang 2004): close / trailing-252 high.
    Stocks near their 52w high keep outperforming — an anchoring effect that
    is largely independent of raw trailing return.
  - 200-DMA regime: context flag (trend filter), not part of the score.
  - Composite = 0.6·z(12-1 return) + 0.4·z(52w-high proximity), winsorized at
    ±3σ so a handful of microcap moonshots don't dominate the cross-section.

Output: data/momentum_signal.parquet  (raw_mom column feeds MasterScore)
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "quant_signals.duckdb"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Lookbacks in TRADING days
SKIP = 21      # ~1 month skip (reversal guard)
LB_6M = 126
LB_12M = 252
MIN_DAYS = 252  # need a full year to rank a stock


def _zwins(s: pd.Series, clip: float = 3.0) -> pd.Series:
    """Cross-sectional z-score, winsorized at +/- clip sigma."""
    mu, sigma = s.mean(), s.std()
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sigma).clip(-clip, clip)


def _find_anchor_date(con, max_gap_days: int = 10):
    """Find the last date of the most recent CONTINUOUS block of >= MIN_DAYS
    trading days. equity_daily can contain isolated stray days separated from
    the dense history by a long ingestion gap; anchoring at the end of the last
    well-populated block keeps every lookback comparison gap-free."""
    dates = con.execute(
        "SELECT DISTINCT trade_date FROM adjusted_prices ORDER BY trade_date"
    ).fetchdf()["trade_date"]
    dates = pd.to_datetime(dates).reset_index(drop=True)
    gap = dates.diff().dt.days.fillna(0)
    seg_id = (gap > max_gap_days).cumsum()
    # pick the latest segment that is at least MIN_DAYS long
    sizes = seg_id.value_counts()
    valid = sorted([s for s in sizes.index if sizes[s] >= MIN_DAYS])
    if not valid:
        return dates.iloc[-1]
    last_seg = valid[-1]
    return dates[seg_id == last_seg].iloc[-1]


def compute_momentum(
    con: Optional[duckdb.DuckDBPyConnection] = None,
    min_turnover: float = 100.0,  # equity_daily.turnover is in LAKH; 100 lakh = Rs 1cr/day
) -> pd.DataFrame:
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        should_close = True
    else:
        should_close = False

    # Anchor at the end of the last continuous block (robust to ingestion gaps).
    anchor = _find_anchor_date(con)
    anchor_str = pd.Timestamp(anchor).date().isoformat()
    print(f"  Anchor date (last continuous block): {anchor_str}")

    # ~420 calendar days ≈ 280 trading days: enough for 252d lookback + buffer.
    # adj_close is split/bonus-adjusted (Total Return Index engine) so a corporate
    # action no longer reads as a phantom crash in the momentum lookbacks.
    # canonical_symbol groups the series by ISIN identity (bridges ticker renames);
    # adj_close is corporate-action adjusted (no phantom split/bonus/demerger crashes).
    df = con.execute(f"""
        SELECT trade_date, canonical_symbol AS symbol, adj_close AS close, turnover
        FROM adjusted_prices
        WHERE adj_close > 0
          AND trade_date <= DATE '{anchor_str}'
          AND trade_date >= DATE '{anchor_str}' - INTERVAL 420 DAY
        ORDER BY canonical_symbol, trade_date
    """).fetchdf()

    if should_close:
        con.close()

    if df.empty:
        return pd.DataFrame()

    results = []
    for symbol, g in df.groupby("symbol"):
        close = g["close"].to_numpy(dtype=float)
        turn = g["turnover"].to_numpy(dtype=float)
        n = len(close)
        if n < LB_12M + 1:  # need t-252 close for the 12-1 lookback
            continue

        c0 = close[-1]
        c_skip = close[-(SKIP + 1)]
        c_6m = close[-(LB_6M + 1)]
        c_12m = close[-(LB_12M + 1)]

        ret_12_1 = c_skip / c_12m - 1.0
        ret_6_1 = c_skip / c_6m - 1.0
        ret_1m = c0 / c_skip - 1.0

        high_252 = close[-LB_12M:].max()
        pct_52w_high = c0 / high_252 if high_252 > 0 else np.nan

        sma200 = close[-200:].mean()
        dist_200dma = c0 / sma200 - 1.0 if sma200 > 0 else np.nan

        avg_turnover_20 = np.nanmean(turn[-20:])

        results.append({
            "symbol": symbol,
            "close": c0,
            "ret_12_1": ret_12_1,
            "ret_6_1": ret_6_1,
            "ret_1m": ret_1m,
            "pct_52w_high": pct_52w_high,
            "dist_200dma": dist_200dma,
            "above_200dma": bool(c0 >= sma200),
            "avg_turnover_20": avg_turnover_20,
        })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    # Liquidity gate: keep illiquid names out of the cross-section ranking
    liquid = out["avg_turnover_20"] >= min_turnover

    z_ret = _zwins(out.loc[liquid, "ret_12_1"])
    z_hi = _zwins(out.loc[liquid, "pct_52w_high"])
    out.loc[liquid, "mom_score"] = 0.6 * z_ret + 0.4 * z_hi
    out["mom_rank"] = out["mom_score"].rank(pct=True)

    out["as_of"] = anchor_str
    out = out.sort_values("mom_score", ascending=False, na_position="last").reset_index(drop=True)
    n_ranked = out["mom_score"].notna().sum()
    print(f"  Momentum computed for {len(out)} symbols ({n_ranked} liquid & ranked), as of {anchor_str}")
    return out


if __name__ == "__main__":
    mom = compute_momentum()
    if not mom.empty:
        ranked = mom[mom["mom_score"].notna()]
        cols = ["symbol", "ret_12_1", "ret_6_1", "pct_52w_high", "above_200dma", "mom_score", "mom_rank"]
        print(f"\n=== Momentum Top 20 ===")
        print(ranked.head(20)[cols].to_string(index=False))
        print(f"\n=== Bottom 10 ===")
        print(ranked.tail(10)[cols].to_string(index=False))
        print(f"\nmom_score stats:")
        print(ranked["mom_score"].describe())

        out = DATA_DIR / "momentum_signal.parquet"
        mom.to_parquet(out, index=False)
        print(f"\nSaved {len(mom)} rows to {out.name}")
