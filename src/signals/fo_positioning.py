"""
Signal 2: F&O Informed Positioning — vectorized

Pan & Poteshman (2006): option order flow predicts stock returns.
OI-price regime + PCR + participant divergence.

All heavy lifting in DuckDB window functions.
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "quant_signals.duckdb"


def compute_stock_fo_signals(
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """Stock-level F&O signals — single DuckDB query with window functions."""
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        should_close = True
    else:
        should_close = False

    print("Computing F&O signals (vectorized)...")

    fo = con.execute("""
        WITH near_month_futures AS (
            SELECT trade_date, symbol, close, oi, oi_change,
                   ROW_NUMBER() OVER (PARTITION BY trade_date, symbol ORDER BY expiry) as rn
            FROM fo_daily
            WHERE instrument = 'FUTSTK' AND oi > 0
        ),
        futures AS (
            SELECT trade_date, symbol, close, oi, oi_change
            FROM near_month_futures WHERE rn = 1
        ),

        -- Stock option OI aggregated by type
        options AS (
            SELECT trade_date, symbol,
                   SUM(CASE WHEN option_type = 'CE' THEN oi ELSE 0 END) as call_oi,
                   SUM(CASE WHEN option_type = 'PE' THEN oi ELSE 0 END) as put_oi
            FROM fo_daily
            WHERE instrument = 'OPTSTK' AND option_type IN ('CE','PE') AND oi > 0
            GROUP BY trade_date, symbol
        ),

        -- Futures with lagged price for regime classification
        futures_enriched AS (
            SELECT
                f.trade_date, f.symbol, f.close, f.oi, f.oi_change,
                f.close / NULLIF(LAG(f.close) OVER (PARTITION BY f.symbol ORDER BY f.trade_date), 0) - 1
                    as price_change,
                -- PCR
                LEAST(GREATEST(o.put_oi::DOUBLE / NULLIF(o.call_oi, 0), 0.1), 5.0) as pcr_oi,
                -- OI regime
                CASE
                    WHEN f.close > LAG(f.close) OVER (PARTITION BY f.symbol ORDER BY f.trade_date)
                         AND f.oi_change > 0 THEN 'long_buildup'
                    WHEN f.close < LAG(f.close) OVER (PARTITION BY f.symbol ORDER BY f.trade_date)
                         AND f.oi_change > 0 THEN 'short_buildup'
                    WHEN f.close > LAG(f.close) OVER (PARTITION BY f.symbol ORDER BY f.trade_date)
                         AND f.oi_change <= 0 THEN 'short_covering'
                    ELSE 'long_unwinding'
                END as oi_regime
            FROM futures f
            LEFT JOIN options o ON f.trade_date = o.trade_date AND f.symbol = o.symbol
        ),

        -- Window-function based rolling stats
        signals AS (
            SELECT
                trade_date, symbol, close, oi, oi_change, price_change,
                oi_regime, pcr_oi,

                -- 5-day OI momentum
                SUM(oi_change) OVER w5 / NULLIF(AVG(oi) OVER w20, 0) as oi_momentum,

                -- PCR z-score (20-day)
                (pcr_oi - AVG(pcr_oi) OVER w20) / NULLIF(STDDEV(pcr_oi) OVER w20, 0) as pcr_zscore,

                -- Regime score: long_buildup=2, short_covering=1, long_unwinding=-1, short_buildup=-2
                AVG(CASE oi_regime
                    WHEN 'long_buildup' THEN 2
                    WHEN 'short_covering' THEN 1
                    WHEN 'long_unwinding' THEN -1
                    WHEN 'short_buildup' THEN -2
                END) OVER w5 as regime_5d

            FROM futures_enriched
            WINDOW
                w5 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
                w20 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
        )

        SELECT
            trade_date, symbol, close, oi, oi_change,
            oi_regime, pcr_oi, oi_momentum, pcr_zscore, regime_5d,
            -- Composite bullish score
            COALESCE(regime_5d, 0) * 0.4
            + COALESCE(oi_momentum, 0) * 0.3
            + COALESCE(pcr_zscore, 0) * 0.3 as fo_bullish_raw
        FROM signals
        ORDER BY symbol, trade_date
    """).fetchdf()

    fo["trade_date"] = pd.to_datetime(fo["trade_date"])
    print(f"  {len(fo)} stock-day records, {fo['symbol'].nunique()} symbols")

    if should_close:
        con.close()

    return fo


def compute_participant_divergence(
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """FII vs Retail divergence — single SQL query."""
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        should_close = True
    else:
        should_close = False

    div = con.execute("""
        WITH pivoted AS (
            SELECT
                trade_date,
                MAX(CASE WHEN client_type = 'FII' THEN total_long END) as fii_long,
                MAX(CASE WHEN client_type = 'FII' THEN total_short END) as fii_short,
                MAX(CASE WHEN client_type = 'Client' THEN total_long END) as client_long,
                MAX(CASE WHEN client_type = 'Client' THEN total_short END) as client_short
            FROM participant_oi
            GROUP BY trade_date
        )
        SELECT
            trade_date,
            (fii_long - fii_short)::DOUBLE / NULLIF(fii_long + fii_short, 0) as fii_bias,
            (client_long - client_short)::DOUBLE / NULLIF(client_long + client_short, 0) as client_bias,
            (fii_long - fii_short)::DOUBLE / NULLIF(fii_long + fii_short, 0)
            - (client_long - client_short)::DOUBLE / NULLIF(client_long + client_short, 0) as divergence,
            fii_long, fii_short, client_long, client_short
        FROM pivoted
        ORDER BY trade_date
    """).fetchdf()

    div["trade_date"] = pd.to_datetime(div["trade_date"])
    print(f"  Participant divergence: {len(div)} days")

    if should_close:
        con.close()

    return div


def compute_pre_earnings_fo(
    con: duckdb.DuckDBPyConnection,
    earnings_df: pd.DataFrame,
    signal_window: tuple = (-10, -1),
) -> pd.DataFrame:
    """Pre-earnings F&O signals — vectorized merge instead of per-event loop."""
    fo = compute_stock_fo_signals(con)

    # For each earnings event, find signal window trading days
    # Use merge_asof to find anchor date, then filter
    earnings_df = earnings_df.copy()
    earnings_df["earnings_date"] = pd.to_datetime(earnings_df["earnings_date"])

    results = []
    for symbol in earnings_df["symbol"].unique():
        sym_fo = fo[fo["symbol"] == symbol].sort_values("trade_date").reset_index(drop=True)
        sym_earn = earnings_df[earnings_df["symbol"] == symbol]

        if len(sym_fo) < 15:
            continue

        dates = sym_fo["trade_date"].values

        for _, row in sym_earn.iterrows():
            edate = np.datetime64(row["earnings_date"])
            mask = dates < edate
            if mask.sum() < 10:
                continue

            anchor = mask.sum() - 1  # index of day -1
            start = max(0, anchor + signal_window[0] + 1)
            end = anchor + signal_window[1] + 1

            window = sym_fo.iloc[start:end + 1]
            if len(window) < 5:
                continue

            results.append({
                "symbol": symbol,
                "earnings_date": row["earnings_date"],
                "mean_regime_5d": window["regime_5d"].mean(),
                "mean_oi_momentum": window["oi_momentum"].mean(),
                "mean_pcr_zscore": window["pcr_zscore"].mean(),
                "fo_composite": window["fo_bullish_raw"].mean(),
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH), read_only=True)

    import time
    t0 = time.time()
    fo = compute_stock_fo_signals(con)
    t1 = time.time()
    print(f"  Completed in {t1-t0:.1f}s")

    print(f"\nOI Regime Distribution:")
    print(fo["oi_regime"].value_counts())
    print(f"\nBullish Score: {fo['fo_bullish_raw'].describe()}")

    t0 = time.time()
    div = compute_participant_divergence(con)
    t1 = time.time()
    print(f"\n  Divergence computed in {t1-t0:.1f}s")
    print(f"  Latest: {div.iloc[-1]['divergence']:+.4f} ({div.iloc[-1]['trade_date'].strftime('%Y-%m-%d')})")

    fo.to_parquet(Path(__file__).resolve().parents[2] / "data" / "fo_signals.parquet", index=False)
    div.to_parquet(Path(__file__).resolve().parents[2] / "data" / "participant_divergence.parquet", index=False)
    con.close()
