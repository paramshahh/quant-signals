"""
Signal 1: Cumulative Abnormal Delivery Percentage (CADP) — vectorized

Kyle (1985) informed trading → delivery % proxies informed order flow.
CADP_i = sum of (DP_{i,t} - baseline_DP_i) over [-10, -1] trading days before earnings.

Vectorized: DuckDB window functions for baseline + signal windows,
numpy for cross-sectional ranking and CAR computation.
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "quant_signals.duckdb"


def compute_cadp_signal(con: Optional[duckdb.DuckDBPyConnection] = None) -> pd.DataFrame:
    """Full CADP pipeline — all computation in DuckDB SQL + vectorized pandas."""
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        should_close = True
    else:
        should_close = False

    print("Computing CADP (vectorized)...")

    # Step 1: Get earnings dates and assign trading-day indices per symbol
    # Step 2: For each earnings event, compute baseline and signal windows using window functions
    # All in one big SQL query
    cadp = con.execute("""
        WITH earnings AS (
            SELECT DISTINCT symbol, meeting_date as earnings_date
            FROM board_meetings
            WHERE purpose LIKE '%Financial Result%'
              AND meeting_date IS NOT NULL
        ),

        -- Assign row numbers (trading day index) per symbol
        equity_numbered AS (
            SELECT
                trade_date, symbol, close, volume, delivery_pct,
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trade_date) as day_idx
            FROM equity_daily
            WHERE series = 'EQ'
              AND delivery_pct IS NOT NULL
              AND delivery_pct > 0
              AND volume > 0
        ),

        -- For each earnings event, find the last trading day before earnings
        earnings_anchor AS (
            SELECT
                e.symbol, e.earnings_date,
                MAX(eq.day_idx) as anchor_idx,
                MAX(eq.trade_date) as anchor_date
            FROM earnings e
            JOIN equity_numbered eq
              ON e.symbol = eq.symbol
             AND eq.trade_date < e.earnings_date
            GROUP BY e.symbol, e.earnings_date
        ),

        -- Join back to get baseline window [-60,-11] and signal window [-10,-1]
        baseline_stats AS (
            SELECT
                ea.symbol, ea.earnings_date, ea.anchor_idx,
                AVG(eq.delivery_pct) as baseline_dp,
                AVG(eq.volume) as baseline_vol
            FROM earnings_anchor ea
            JOIN equity_numbered eq
              ON ea.symbol = eq.symbol
             AND eq.day_idx BETWEEN (ea.anchor_idx - 59) AND (ea.anchor_idx - 10)
            WHERE ea.anchor_idx >= 60
            GROUP BY ea.symbol, ea.earnings_date, ea.anchor_idx
            HAVING COUNT(*) >= 30
        ),

        signal_stats AS (
            SELECT
                ea.symbol, ea.earnings_date,
                AVG(eq.delivery_pct) as mean_signal_dp,
                SUM(eq.delivery_pct) as sum_signal_dp,
                AVG(eq.volume) as signal_vol,
                COUNT(*) as signal_days,
                -- Pre-return: use first and last close in signal window
                LAST(eq.close ORDER BY eq.day_idx) as close_last,
                FIRST(eq.close ORDER BY eq.day_idx) as close_first
            FROM earnings_anchor ea
            JOIN equity_numbered eq
              ON ea.symbol = eq.symbol
             AND eq.day_idx BETWEEN (ea.anchor_idx - 9) AND ea.anchor_idx
            WHERE ea.anchor_idx >= 60
            GROUP BY ea.symbol, ea.earnings_date
            HAVING COUNT(*) >= 5
        ),

        -- CADP = sum(signal_dp) - signal_days * baseline_dp
        cadp_raw AS (
            SELECT
                b.symbol, b.earnings_date,
                s.sum_signal_dp - (s.signal_days * b.baseline_dp) as cadp,
                b.baseline_dp,
                s.mean_signal_dp,
                s.signal_days,
                (s.close_last / s.close_first) - 1 as pre_return,
                s.signal_vol / NULLIF(b.baseline_vol, 0) as pre_volume_ratio
            FROM baseline_stats b
            JOIN signal_stats s
              ON b.symbol = s.symbol
             AND b.earnings_date = s.earnings_date
        )

        SELECT * FROM cadp_raw
        ORDER BY symbol, earnings_date
    """).fetchdf()

    print(f"  {len(cadp)} stock-earnings events scored")

    if cadp.empty:
        if should_close:
            con.close()
        return cadp

    cadp["earnings_date"] = pd.to_datetime(cadp["earnings_date"])

    # Step 3: Compute post-earnings CAR[0,+5] — vectorized via DuckDB
    # Build a returns table first, then join
    car = con.execute("""
        WITH earnings AS (
            SELECT DISTINCT symbol, meeting_date as earnings_date
            FROM board_meetings
            WHERE purpose LIKE '%Financial Result%'
              AND meeting_date IS NOT NULL
        ),

        -- Daily returns
        returns AS (
            SELECT
                trade_date, symbol, close,
                close / LAG(close) OVER (PARTITION BY symbol ORDER BY trade_date) - 1 as ret
            FROM equity_daily
            WHERE series = 'EQ' AND close > 0
        ),

        -- Market return (equal-weight)
        market_ret AS (
            SELECT trade_date, AVG(ret) as mkt_ret
            FROM returns
            WHERE ret IS NOT NULL
            GROUP BY trade_date
        ),

        -- For each earnings event, get post-earnings trading days
        post_days AS (
            SELECT
                e.symbol, e.earnings_date,
                r.trade_date, r.close, r.ret,
                ROW_NUMBER() OVER (
                    PARTITION BY e.symbol, e.earnings_date
                    ORDER BY r.trade_date
                ) as post_day_num
            FROM earnings e
            JOIN returns r
              ON e.symbol = r.symbol
             AND r.trade_date >= e.earnings_date
        ),

        -- Get close at day 0 and day 5
        post_returns AS (
            SELECT
                p.symbol, p.earnings_date,
                MAX(CASE WHEN post_day_num = 1 THEN close END) as close_0,
                MAX(CASE WHEN post_day_num = 6 THEN close END) as close_5
            FROM post_days p
            WHERE post_day_num <= 6
            GROUP BY p.symbol, p.earnings_date
            HAVING close_0 IS NOT NULL AND close_5 IS NOT NULL
        ),

        -- Market cumulative return over same 5-day window
        post_market AS (
            SELECT
                p.symbol, p.earnings_date,
                EXP(SUM(LN(1 + m.mkt_ret))) - 1 as mkt_cum_ret
            FROM post_days p
            JOIN market_ret m ON p.trade_date = m.trade_date
            WHERE p.post_day_num BETWEEN 2 AND 6
              AND m.mkt_ret IS NOT NULL
            GROUP BY p.symbol, p.earnings_date
        )

        SELECT
            pr.symbol, pr.earnings_date,
            (pr.close_5 / pr.close_0) - 1 as stock_ret_5d,
            COALESCE(pm.mkt_cum_ret, 0) as mkt_ret_5d,
            (pr.close_5 / pr.close_0) - 1 - COALESCE(pm.mkt_cum_ret, 0) as car_0_5
        FROM post_returns pr
        LEFT JOIN post_market pm
          ON pr.symbol = pm.symbol
         AND pr.earnings_date = pm.earnings_date
        ORDER BY pr.symbol, pr.earnings_date
    """).fetchdf()

    car["earnings_date"] = pd.to_datetime(car["earnings_date"])

    # Merge CADP with CAR
    cadp = cadp.merge(
        car[["symbol", "earnings_date", "car_0_5"]],
        on=["symbol", "earnings_date"],
        how="left",
    )

    valid = cadp["car_0_5"].notna().sum()
    print(f"  {valid} events with valid CAR[0,+5]")

    if should_close:
        con.close()

    return cadp


def quintile_analysis(cadp_df: pd.DataFrame) -> pd.DataFrame:
    df = cadp_df.dropna(subset=["cadp", "car_0_5"]).copy()
    df["quintile"] = pd.qcut(df["cadp"], 5, labels=[1, 2, 3, 4, 5])

    summary = df.groupby("quintile", observed=False).agg(
        n=("car_0_5", "count"),
        mean_cadp=("cadp", "mean"),
        mean_car=("car_0_5", "mean"),
        median_car=("car_0_5", "median"),
        std_car=("car_0_5", "std"),
        mean_pre_return=("pre_return", "mean"),
        mean_vol_ratio=("pre_volume_ratio", "mean"),
    ).reset_index()

    summary["t_stat"] = summary["mean_car"] / (summary["std_car"] / np.sqrt(summary["n"]))

    q5 = df[df["quintile"] == 5]["car_0_5"]
    q1 = df[df["quintile"] == 1]["car_0_5"]
    spread = q5.mean() - q1.mean()
    spread_se = np.sqrt(q5.var() / len(q5) + q1.var() / len(q1))
    spread_t = spread / spread_se if spread_se > 0 else 0

    print(f"\nCADP Quintile Analysis (Q5-Q1 spread: {spread:.4f}, t={spread_t:.2f})")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    cadp = compute_cadp_signal()
    if not cadp.empty:
        quintile_analysis(cadp)
        out = Path(__file__).resolve().parents[2] / "data" / "cadp_signal.parquet"
        cadp.to_parquet(out, index=False)
        print(f"\nSaved {len(cadp)} events to {out.name}")
