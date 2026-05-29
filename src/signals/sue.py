"""
Signal 5: Standardized Unexpected Earnings (SUE)

Based on Param's Momentum-Research implementation (Novy-Marx 2013 proxy).
SUE = (EPS_q - EPS_{q-4}) / std(sequential EPS changes over 8 quarters)

Uses screener.in quarterly EPS data from the quarterly_results table.
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "quant_signals.duckdb"


def _calc_sue(eps_array: np.ndarray) -> float:
    """Core SUE calculation — direct port from Momentum-Research/Rank_SUE.ipynb."""
    if len(eps_array) < 8:
        return np.nan
    arr = eps_array[:8]
    if np.any(np.isnan(arr)):
        return np.nan

    yoy = arr[0] - arr[4]
    mean_change = (arr[0] - arr[7]) / 7
    dev = sum(((arr[i] - arr[i + 1]) - mean_change) ** 2 for i in range(7))
    std = (dev / 7) ** 0.5

    if std == 0:
        return np.nan
    return yoy / std


def compute_sue(
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        should_close = True
    else:
        should_close = False

    quarterly = con.execute("""
        SELECT symbol, quarter, eps
        FROM quarterly_results
        WHERE eps IS NOT NULL
        ORDER BY symbol, quarter DESC
    """).fetchdf()

    if should_close:
        con.close()

    quarterly["quarter"] = pd.to_datetime(quarterly["quarter"])

    results = []
    for symbol, group in quarterly.groupby("symbol"):
        g = group.sort_values("quarter", ascending=False).reset_index(drop=True)
        if len(g) < 8:
            continue

        eps_arr = g["eps"].values.astype(float)
        sue = _calc_sue(eps_arr)
        if np.isnan(sue):
            continue

        # Also compute fundamental momentum (Novy-Marx proxy)
        if len(g) >= 8:
            recent4 = eps_arr[:4].mean()
            prior4 = eps_arr[4:8].mean()
            fm_score = recent4 - prior4
        else:
            fm_score = np.nan

        results.append({
            "symbol": symbol,
            "latest_quarter": g.iloc[0]["quarter"],
            "eps_latest": eps_arr[0],
            "eps_yoy_ago": eps_arr[4] if len(eps_arr) > 4 else np.nan,
            "sue": sue,
            "fm_score": fm_score,
            "n_quarters": len(g),
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    df["sue_rank"] = df["sue"].rank(pct=True)
    df["fm_rank"] = df["fm_score"].rank(pct=True)

    df = df.sort_values("sue", ascending=False).reset_index(drop=True)
    print(f"  SUE computed for {len(df)} symbols")
    return df


def compute_sue_history(
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """Compute SUE for every quarter where we have 8+ quarters of history."""
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        should_close = True
    else:
        should_close = False

    quarterly = con.execute("""
        SELECT symbol, quarter, eps
        FROM quarterly_results
        WHERE eps IS NOT NULL
        ORDER BY symbol, quarter
    """).fetchdf()

    if should_close:
        con.close()

    quarterly["quarter"] = pd.to_datetime(quarterly["quarter"])

    results = []
    for symbol, group in quarterly.groupby("symbol"):
        g = group.sort_values("quarter").reset_index(drop=True)
        if len(g) < 8:
            continue

        for i in range(7, len(g)):
            window = g.iloc[i - 7:i + 1]["eps"].values.astype(float)[::-1]
            sue = _calc_sue(window)
            if not np.isnan(sue):
                results.append({
                    "symbol": symbol,
                    "quarter": g.iloc[i]["quarter"],
                    "sue": sue,
                })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(["symbol", "quarter"]).reset_index(drop=True)
        print(f"  SUE history: {len(df)} symbol-quarter observations")
    return df


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH), read_only=True)

    sue = compute_sue(con)
    if not sue.empty:
        print(f"\n=== SUE Rankings (top 20) ===")
        print(sue.head(20)[["symbol", "latest_quarter", "eps_latest", "sue", "sue_rank"]].to_string(index=False))

        print(f"\n=== Bottom 10 ===")
        print(sue.tail(10)[["symbol", "latest_quarter", "eps_latest", "sue", "sue_rank"]].to_string(index=False))

        print(f"\n=== Fundamental Momentum (top 20) ===")
        fm_sorted = sue.sort_values("fm_score", ascending=False)
        print(fm_sorted.head(20)[["symbol", "fm_score", "fm_rank"]].to_string(index=False))

        print(f"\nSUE stats:")
        print(sue["sue"].describe())

        out = Path(__file__).resolve().parents[2] / "data" / "sue_signal.parquet"
        sue.to_parquet(out, index=False)
        print(f"\nSaved {len(sue)} rows to {out.name}")

    con.close()
