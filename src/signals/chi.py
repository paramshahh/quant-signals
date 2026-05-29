"""
Signal 3: Corporate Health Index (CHI) — vectorized

Jeng/Metrick/Zeckhauser (2003): insider buying predicts (selling doesn't).
CHI = f(promoter_delta, insider_net_buy, pledge_activity)

DuckDB for promoter deltas + insider aggregation, vectorized pandas for composite.
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "quant_signals.duckdb"


def _build_crosswalk(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Build BSE security_name → NSE symbol mapping via DuckDB."""
    return con.execute("""
        WITH nse AS (
            SELECT DISTINCT symbol,
                   LOWER(REPLACE(REPLACE(REPLACE(REPLACE(
                       company_name, ' Limited', ''), ' Ltd', ''),
                       '.', ''), '&', 'and')) as norm_name
            FROM promoter_shareholding
            WHERE company_name IS NOT NULL
        ),
        bse AS (
            SELECT DISTINCT security_name,
                   LOWER(REPLACE(REPLACE(REPLACE(REPLACE(
                       security_name, ' Limited', ''), ' Ltd', ''),
                       '.', ''), '&', 'and')) as norm_name
            FROM insider_trading
            WHERE security_name IS NOT NULL
        )
        SELECT b.security_name, n.symbol
        FROM bse b
        JOIN nse n ON b.norm_name = n.norm_name
    """).fetchdf()


def compute_chi(
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        should_close = True
    else:
        should_close = False

    print("Computing CHI (vectorized)...")

    # Promoter deltas — all in SQL
    deltas = con.execute("""
        SELECT
            symbol, quarter_end, promoter_pct,
            promoter_pct - LAG(promoter_pct) OVER (
                PARTITION BY symbol ORDER BY quarter_end
            ) as delta_ps,
            LAG(quarter_end) OVER (
                PARTITION BY symbol ORDER BY quarter_end
            ) as prev_quarter
        FROM promoter_shareholding
        WHERE promoter_pct IS NOT NULL
        ORDER BY symbol, quarter_end
    """).fetchdf()

    deltas = deltas.dropna(subset=["delta_ps"])
    deltas["quarter_end"] = pd.to_datetime(deltas["quarter_end"])
    deltas["prev_quarter"] = pd.to_datetime(deltas["prev_quarter"])
    deltas = deltas[
        (deltas["quarter_end"] - deltas["prev_quarter"]).dt.days <= 200
    ].drop(columns=["prev_quarter"])

    print(f"  {len(deltas)} promoter delta observations, {deltas['symbol'].nunique()} symbols")

    # BSE→NSE crosswalk
    crosswalk = _build_crosswalk(con)
    print(f"  {len(crosswalk)} BSE→NSE mappings")

    # Insider activity — aggregated per symbol per quarter, all in SQL
    # First register the crosswalk as a temp table
    con.register("crosswalk_df", crosswalk)

    insider_quarterly = con.execute("""
        WITH mapped AS (
            SELECT
                c.symbol,
                i.transaction_type,
                i.acquired_value,
                i.date_from,
                -- Align to quarter end
                MAKE_DATE(
                    YEAR(i.date_from) + CASE WHEN MONTH(i.date_from) > 3 THEN 0 ELSE -1 END,
                    CASE
                        WHEN MONTH(i.date_from) BETWEEN 4 AND 6 THEN 6
                        WHEN MONTH(i.date_from) BETWEEN 7 AND 9 THEN 9
                        WHEN MONTH(i.date_from) BETWEEN 10 AND 12 THEN 12
                        ELSE 3
                    END,
                    CASE
                        WHEN MONTH(i.date_from) BETWEEN 4 AND 6 THEN 30
                        WHEN MONTH(i.date_from) BETWEEN 7 AND 9 THEN 30
                        WHEN MONTH(i.date_from) BETWEEN 10 AND 12 THEN 31
                        ELSE 31
                    END
                ) as quarter_end
            FROM insider_trading i
            JOIN crosswalk_df c ON i.security_name = c.security_name
            WHERE i.date_from IS NOT NULL
              AND i.category IN ('Promoter','Promoter & Director','Promoter Group',
                                 'Director','KMP','Designated Person')
        )
        SELECT
            symbol,
            quarter_end,
            SUM(CASE WHEN transaction_type = 'Acquisition' THEN COALESCE(acquired_value, 0) ELSE 0 END)
            - SUM(CASE WHEN transaction_type = 'Disposal' THEN COALESCE(acquired_value, 0) ELSE 0 END)
            as net_buy_value,
            SUM(CASE WHEN transaction_type = 'Pledge' THEN 1 ELSE 0 END)
            - SUM(CASE WHEN transaction_type = 'Revoke' THEN 1 ELSE 0 END)
            as pledge_net,
            COUNT(*) as total_txns
        FROM mapped
        GROUP BY symbol, quarter_end
        ORDER BY symbol, quarter_end
    """).fetchdf()

    insider_quarterly["quarter_end"] = pd.to_datetime(insider_quarterly["quarter_end"])
    print(f"  {len(insider_quarterly)} insider quarterly aggregates, {insider_quarterly['symbol'].nunique()} symbols")

    # Merge promoter deltas + insider activity
    merged = deltas.merge(
        insider_quarterly,
        on=["symbol", "quarter_end"],
        how="left",
    )
    merged["net_buy_value"] = merged["net_buy_value"].fillna(0)
    merged["pledge_net"] = merged["pledge_net"].fillna(0)

    # Cross-sectional ranking per quarter — fully vectorized (no groupby.apply, which in
    # recent pandas drops the grouping column from the result and broke the downstream sort).
    g = merged.copy()
    grp = g.groupby("quarter_end")
    q_count = grp["symbol"].transform("count")

    # Asymmetric delta: promoter buying gets full weight, selling 0.5x
    g["_delta_adj"] = np.where(g["delta_ps"] > 0, g["delta_ps"], g["delta_ps"] * 0.5)
    rank_delta = g.groupby("quarter_end")["_delta_adj"].rank(pct=True)
    rank_buy = grp["net_buy_value"].rank(pct=True)
    rank_pledge = grp["pledge_net"].rank(pct=True)

    chi_raw = 0.4 * rank_delta + 0.35 * rank_buy - 0.25 * rank_pledge
    # Rescale to [0, 1] within each quarter
    cmin = chi_raw.groupby(g["quarter_end"]).transform("min")
    cmax = chi_raw.groupby(g["quarter_end"]).transform("max")
    chi_scaled = np.where(cmax > cmin, (chi_raw - cmin) / (cmax - cmin), chi_raw)

    # require >= 10 names in the quarter for a meaningful cross-sectional rank
    g["chi"] = np.where(q_count >= 10, chi_scaled, np.nan)
    result = g.drop(columns=["_delta_adj"]).dropna(subset=["chi"]).sort_values(["symbol", "quarter_end"])

    print(f"  {len(result)} CHI scores computed")

    if should_close:
        con.close()

    return result[["symbol", "quarter_end", "promoter_pct", "delta_ps",
                    "net_buy_value", "pledge_net", "chi"]]


def decile_analysis(chi_df: pd.DataFrame) -> pd.DataFrame:
    df = chi_df.dropna(subset=["chi"]).copy()
    df["decile"] = pd.qcut(df["chi"], 10, labels=range(1, 11), duplicates="drop")

    summary = df.groupby("decile", observed=False).agg(
        n=("chi", "count"),
        mean_chi=("chi", "mean"),
        mean_delta_ps=("delta_ps", "mean"),
        mean_net_buy=("net_buy_value", "mean"),
        mean_pledge_net=("pledge_net", "mean"),
    ).reset_index()

    print("\nCHI Decile Summary (10 = healthiest)")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    chi = compute_chi()
    if not chi.empty:
        print(f"\n{len(chi)} observations, {chi['symbol'].nunique()} symbols, {chi['quarter_end'].nunique()} quarters")
        print(chi["chi"].describe())
        decile_analysis(chi)
        chi.to_parquet(Path(__file__).resolve().parents[2] / "data" / "chi_signal.parquet", index=False)
