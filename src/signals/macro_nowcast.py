"""
Signal 4: Macro Sector Revenue Nowcaster (vectorized)

Giannone/Reichlin/Small (2008) nowcasting framework.
GST + IIP + RBI credit → macro momentum composite.

All computation via DuckDB SQL — no Python loops.
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "quant_signals.duckdb"

IIP_SECTOR_MAP = {
    "General": "Market",
    "Mining": "Mining",
    "Electricity": "Power",
    "Manufacture of Basic Metals": "Metals",
    "Manufacture of Chemicals and Chemical Products": "Chemicals",
    "Manufacture of Pharmaceuticals, Medicinal Chemical and Botan...": "Pharma",
    "Manufacture of Motor Vehicles, Trailers and Semi-trailers": "Auto",
    "Manufacture of Food Products": "FMCG",
    "Manufacture of Textiles": "Textiles",
    "Manufacture of Computer, Electronic and Optical Products": "IT Hardware",
    "Manufacture of Electrical Equipment": "Capital Goods",
    "Manufacture of Coke and Refined Petroleum Products": "Oil & Gas",
    "Manufacture of Rubber and Plastics Products": "Chemicals",
    "Manufacture of Other Non-metallic Mineral Products": "Cement",
}


def compute_macro_dashboard(
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> dict:
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        should_close = True
    else:
        should_close = False

    # GST: national aggregate YoY (not mean of state YoYs)
    national_gst = con.execute("""
        WITH monthly_national AS (
            SELECT month, SUM(total_cr) as total_cr
            FROM gst_collections
            GROUP BY month
        )
        SELECT
            c.month,
            c.total_cr,
            p.total_cr as prev_year_cr,
            (c.total_cr / p.total_cr) - 1 as gst_yoy
        FROM monthly_national c
        JOIN monthly_national p
          ON MONTH(c.month) = MONTH(p.month)
         AND YEAR(c.month) = YEAR(p.month) + 1
        WHERE p.total_cr > 0
          AND (c.total_cr / p.total_cr) BETWEEN 0.5 AND 2.0
        ORDER BY c.month
    """).fetchdf()
    national_gst["month"] = pd.to_datetime(national_gst["month"])

    # Top 5 states YoY
    state_gst = con.execute("""
        WITH state_monthly AS (
            SELECT month, state, total_cr
            FROM gst_collections
            WHERE state IN ('Maharashtra','Karnataka','Tamil Nadu','Gujarat','Haryana')
        )
        SELECT
            c.month, c.state,
            (c.total_cr / p.total_cr) - 1 as gst_yoy
        FROM state_monthly c
        JOIN state_monthly p
          ON c.state = p.state
         AND MONTH(c.month) = MONTH(p.month)
         AND YEAR(c.month) = YEAR(p.month) + 1
        WHERE p.total_cr > 100
        ORDER BY c.month, c.state
    """).fetchdf()
    state_gst["month"] = pd.to_datetime(state_gst["month"])
    state_pivot = state_gst.pivot_table(index="month", columns="state", values="gst_yoy").reset_index()

    # IIP: general index YoY
    iip = con.execute("""
        SELECT
            c.period, c.sector, c.index_value,
            (c.index_value / p.index_value) - 1 as iip_yoy
        FROM iip_index c
        JOIN iip_index p
          ON c.sector = p.sector
         AND MONTH(c.period) = MONTH(p.period)
         AND YEAR(c.period) = YEAR(p.period) + 1
        WHERE p.index_value > 0
        ORDER BY c.sector, c.period
    """).fetchdf()
    iip["period"] = pd.to_datetime(iip["period"])

    general_iip = iip[iip["sector"] == "General"][["period", "iip_yoy"]].copy()
    general_iip.columns = ["month", "iip_yoy"]

    # RBI credit: total non-food credit YoY
    credit = con.execute("""
        SELECT report_date, sector, outstanding_cr, yoy_pct / 100.0 as credit_yoy
        FROM rbi_sectoral_credit
        WHERE sector = 'III. Non-food Credit'
          AND outstanding_cr > 0
        ORDER BY report_date
    """).fetchdf()
    credit["report_date"] = pd.to_datetime(credit["report_date"])
    credit_ts = credit[["report_date", "credit_yoy"]].copy()
    credit_ts.columns = ["month", "credit_yoy"]

    # Composite: merge on month
    macro = national_gst[["month", "gst_yoy"]].merge(general_iip, on="month", how="outer")
    macro = macro.merge(credit_ts, on="month", how="outer")
    macro = macro.sort_values("month").reset_index(drop=True)

    # Equal-weight composite
    signals = ["gst_yoy", "iip_yoy", "credit_yoy"]
    macro["macro_momentum"] = macro[signals].mean(axis=1)

    # Z-score relative to trailing 12 months
    macro["momentum_12m_mean"] = macro["macro_momentum"].rolling(12, min_periods=6).mean()
    macro["momentum_12m_std"] = macro["macro_momentum"].rolling(12, min_periods=6).std()
    macro["momentum_zscore"] = (
        (macro["macro_momentum"] - macro["momentum_12m_mean"])
        / macro["momentum_12m_std"]
    )

    if should_close:
        con.close()

    return {
        "national_gst": national_gst,
        "state_gst": state_pivot,
        "iip": iip,
        "macro_monthly": macro,
    }


def compute_sector_momentum(
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        should_close = True
    else:
        should_close = False

    iip = con.execute("""
        WITH latest_month AS (
            SELECT MAX(period) as max_period FROM iip_index
        ),
        yoy AS (
            SELECT
                c.period, c.sector, c.index_value,
                (c.index_value / p.index_value) - 1 as iip_yoy
            FROM iip_index c
            JOIN iip_index p
              ON c.sector = p.sector
             AND MONTH(c.period) = MONTH(p.period)
             AND YEAR(c.period) = YEAR(p.period) + 1
            WHERE p.index_value > 0
        )
        SELECT sector, iip_yoy
        FROM yoy
        WHERE period = (SELECT max_period FROM latest_month)
        ORDER BY iip_yoy DESC
    """).fetchdf()

    if should_close:
        con.close()

    iip["nse_sector"] = iip["sector"].map(IIP_SECTOR_MAP)
    iip = iip.dropna(subset=["nse_sector"])
    iip["iip_rank"] = iip["iip_yoy"].rank(pct=True)
    return iip[["nse_sector", "iip_yoy", "iip_rank"]].rename(columns={"nse_sector": "sector"})


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH), read_only=True)
    result = compute_macro_dashboard(con)

    print("=== National GST YoY (latest 6 months) ===")
    print(result["national_gst"].tail(6).to_string(index=False))

    print("\n=== State GST YoY (latest 3 months) ===")
    print(result["state_gst"].tail(3).to_string(index=False))

    print("\n=== Macro Composite (latest 12 months) ===")
    cols = ["month", "gst_yoy", "iip_yoy", "credit_yoy", "macro_momentum", "momentum_zscore"]
    latest = result["macro_monthly"].dropna(subset=["macro_momentum"]).tail(12)
    print(latest[cols].to_string(index=False))

    print("\n=== Sector IIP Momentum ===")
    sectors = compute_sector_momentum(con)
    print(sectors.to_string(index=False))

    result["macro_monthly"].to_parquet(
        Path(__file__).resolve().parents[2] / "data" / "macro_nowcast.parquet", index=False
    )
    con.close()
