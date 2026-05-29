"""
MasterScore — composite ranking across all signals.

Cross-sectional z-score of each signal, then equal-weight average.
- Signal 1: CADP (pre-earnings delivery conviction)
- Signal 2: F&O positioning (OI regime + PCR composite)
- Signal 3: CHI (corporate health index)
- Signal 4: Macro momentum (sector tilt via IIP)
- Signal 5: SUE (standardized unexpected earnings)
- Signal 7: Price momentum (12-1 return + 52w-high proximity)
- Signal 8: Value + Quality (cheapness + profitability/leverage/margin trend)

Adding price momentum is the important structural change: the composite was
previously all fundamentals/flows with NO price factor. Momentum is the most
robust standalone factor and is largely uncorrelated with the earnings/flow
signals, so it diversifies the composite rather than just padding it.
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "quant_signals.duckdb"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _zscore(s: pd.Series) -> pd.Series:
    mu, sigma = s.mean(), s.std()
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def _load_signal(path: str, symbol_col: str = "symbol", score_col: str = None,
                 latest_only: bool = True, date_col: str = None) -> pd.DataFrame:
    fpath = DATA_DIR / path
    if not fpath.exists():
        print(f"  SKIP: {path} not found")
        return pd.DataFrame()
    df = pd.read_parquet(fpath)
    if latest_only and date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        latest = df[date_col].max()
        df = df[df[date_col] == latest]
    return df


def compute_master_score(
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        should_close = True
    else:
        should_close = False

    signals = {}

    # Signal 1: CADP — use latest earnings event per symbol
    cadp = _load_signal("cadp_signal.parquet")
    if not cadp.empty:
        cadp_latest = cadp.sort_values("earnings_date").groupby("symbol").tail(1)
        signals["cadp"] = cadp_latest[["symbol", "cadp"]].rename(columns={"cadp": "raw_cadp"})
        print(f"  CADP: {len(signals['cadp'])} symbols")

    # Signal 2: F&O — latest day composite
    fo = _load_signal("fo_signals.parquet")
    if not fo.empty:
        fo["trade_date"] = pd.to_datetime(fo["trade_date"])
        latest_fo = fo[fo["trade_date"] == fo["trade_date"].max()]
        signals["fo"] = latest_fo[["symbol", "fo_bullish_raw"]].rename(columns={"fo_bullish_raw": "raw_fo"})
        print(f"  F&O: {len(signals['fo'])} symbols")

    # Signal 3: CHI — latest quarter per symbol
    chi = _load_signal("chi_signal.parquet")
    if not chi.empty:
        chi["quarter_end"] = pd.to_datetime(chi["quarter_end"])
        chi_latest = chi.sort_values("quarter_end").groupby("symbol").tail(1)
        signals["chi"] = chi_latest[["symbol", "chi"]].rename(columns={"chi": "raw_chi"})
        print(f"  CHI: {len(signals['chi'])} symbols")

    # Signal 4: Macro — sector-level tilt (applied per stock via sector mapping)
    # For now, macro is a market-wide signal, not stock-level.
    # We apply it as a uniform tilt (same for all stocks).
    macro = _load_signal("macro_nowcast.parquet")
    if not macro.empty:
        macro["month"] = pd.to_datetime(macro["month"])
        latest_macro = macro.dropna(subset=["momentum_zscore"]).iloc[-1]
        macro_zscore = latest_macro["momentum_zscore"]
        print(f"  Macro: z={macro_zscore:.2f} (uniform tilt)")
    else:
        macro_zscore = 0.0

    # Signal 5: SUE
    sue = _load_signal("sue_signal.parquet")
    if not sue.empty:
        signals["sue"] = sue[["symbol", "sue"]].rename(columns={"sue": "raw_sue"})
        print(f"  SUE: {len(signals['sue'])} symbols")

    # Signal 7: Price momentum (12-1 return + 52w-high proximity)
    mom = _load_signal("momentum_signal.parquet")
    if not mom.empty and "mom_score" in mom.columns:
        mom = mom[mom["mom_score"].notna()]
        signals["mom"] = mom[["symbol", "mom_score"]].rename(columns={"mom_score": "raw_mom"})
        print(f"  Momentum: {len(signals['mom'])} symbols")

    # Signal 8: Value + Quality composite
    vq = _load_signal("value_quality_signal.parquet")
    if not vq.empty and "vq_score" in vq.columns:
        vq = vq[vq["vq_score"].notna()]
        signals["vq"] = vq[["symbol", "vq_score"]].rename(columns={"vq_score": "raw_vq"})
        print(f"  Value+Quality: {len(signals['vq'])} symbols")

    if should_close:
        con.close()

    if not signals:
        print("  No signals available")
        return pd.DataFrame()

    # Merge all signals on symbol
    merged = None
    for name, df in signals.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="symbol", how="outer")

    raw_cols = [c for c in merged.columns if c.startswith("raw_")]
    print(f"\n  Merged: {len(merged)} symbols across {len(raw_cols)} signals")
    print(f"  Coverage: {merged[raw_cols].notna().sum().to_dict()}")

    # Z-score each signal cross-sectionally
    z_cols = []
    for col in raw_cols:
        z_col = col.replace("raw_", "z_")
        merged[z_col] = _zscore(merged[col])
        z_cols.append(z_col)

    # Equal-weight composite (mean of available z-scores per stock)
    merged["master_score"] = merged[z_cols].mean(axis=1)

    # Add macro tilt (small uniform bump based on macro regime)
    merged["master_score_tilted"] = merged["master_score"] + 0.1 * macro_zscore

    # Rank
    merged["master_rank"] = merged["master_score_tilted"].rank(pct=True)
    merged = merged.sort_values("master_score_tilted", ascending=False).reset_index(drop=True)

    # Signal coverage count
    merged["n_signals"] = merged[raw_cols].notna().sum(axis=1).astype(int)

    # Require at least 2 signals for a meaningful composite
    merged.loc[merged["n_signals"] < 2, "master_score_tilted"] = np.nan
    merged["master_rank"] = merged["master_score_tilted"].rank(pct=True)
    merged = merged.sort_values("master_score_tilted", ascending=False).reset_index(drop=True)

    return merged


if __name__ == "__main__":
    ms = compute_master_score()
    if not ms.empty:
        print(f"\n=== MasterScore Top 30 ===")
        display_cols = ["symbol", "master_score_tilted", "master_rank", "n_signals"]
        raw_cols = [c for c in ms.columns if c.startswith("raw_")]
        print(ms.head(30)[display_cols + raw_cols].to_string(index=False))

        print(f"\n=== Bottom 10 ===")
        print(ms.tail(10)[display_cols].to_string(index=False))

        print(f"\n=== Signal coverage ===")
        print(ms["n_signals"].value_counts().sort_index())

        print(f"\nMasterScore stats:")
        print(ms["master_score_tilted"].describe())

        out = DATA_DIR / "master_score.parquet"
        ms.to_parquet(out, index=False)
        print(f"\nSaved {len(ms)} rows to {out.name}")
