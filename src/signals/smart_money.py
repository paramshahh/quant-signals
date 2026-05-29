"""
Signal 9: Smart-Money Flow (bulk & block deals)

Bulk deals (>0.5% of equity in a day) and block deals (negotiated large trades)
are SEBI-mandated same-day disclosures naming the counterparty. They are the
cleanest public read on what large/known investors are actually doing — a tape
of institutional and HNI conviction that the rest of the dashboard never used.

Per symbol over a trailing window we compute:
  - buy_value / sell_value / net_value (Rs cr)
  - n_buys, n_sells, n_unique_buyers
  - repeat_accumulation: a single client appearing as a buyer on 2+ days
    (sticky conviction, not a one-off rebalance)
  - net_z: cross-sectional z of net_value  (sign of smart-money pressure)

NOTE ON COVERAGE: bulk/block ingestion currently holds only a short recent
window, so this is a "who's accumulating right now" tape rather than a deep
backtested factor. It is intentionally NOT folded into MasterScore yet — the
cross-section is too sparse and would inject noise. It graduates into the
composite once the history deepens.

Output: data/smart_money_signal.parquet (per-symbol), data/smart_money_deals.parquet (raw recent deals)
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "quant_signals.duckdb"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

WINDOW_DAYS = 30


def _zwins(s: pd.Series, clip: float = 3.0) -> pd.Series:
    mu, sigma = s.mean(), s.std()
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sigma).clip(-clip, clip)


def compute_smart_money(
    con: Optional[duckdb.DuckDBPyConnection] = None,
    window_days: int = WINDOW_DAYS,
):
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        should_close = True
    else:
        should_close = False

    # Union bulk + block, tag the source, restrict to the trailing window
    deals = con.execute(f"""
        WITH alld AS (
            SELECT trade_date, symbol, security_name, client_name, buy_sell,
                   quantity, price, 'bulk' AS deal_type FROM bulk_deals
            UNION ALL
            SELECT trade_date, symbol, security_name, client_name, buy_sell,
                   quantity, price, 'block' AS deal_type FROM block_deals
        )
        SELECT * FROM alld
        WHERE trade_date >= (SELECT MAX(trade_date) FROM alld) - INTERVAL {window_days} DAY
        ORDER BY trade_date DESC, quantity * price DESC
    """).fetchdf()

    if should_close:
        con.close()

    if deals.empty:
        print("  No bulk/block deals in window")
        return pd.DataFrame(), pd.DataFrame()

    deals["value_cr"] = deals["quantity"] * deals["price"] / 1e7
    deals["side"] = deals["buy_sell"].str.upper().str.strip()

    rows = []
    for symbol, g in deals.groupby("symbol"):
        buys = g[g["side"] == "BUY"]
        sells = g[g["side"] == "SELL"]
        buy_val = buys["value_cr"].sum()
        sell_val = sells["value_cr"].sum()
        # repeat accumulation: same buyer on 2+ distinct days
        repeat = False
        if not buys.empty:
            per_client_days = buys.groupby("client_name")["trade_date"].nunique()
            repeat = bool((per_client_days >= 2).any())
        rows.append({
            "symbol": symbol,
            "security_name": g["security_name"].iloc[0],
            "buy_value_cr": buy_val,
            "sell_value_cr": sell_val,
            "net_value_cr": buy_val - sell_val,
            "n_buys": len(buys),
            "n_sells": len(sells),
            "n_unique_buyers": buys["client_name"].nunique(),
            "repeat_accumulation": repeat,
            "last_date": g["trade_date"].max(),
        })

    agg = pd.DataFrame(rows)
    agg["net_z"] = _zwins(agg["net_value_cr"])
    agg = agg.sort_values("net_value_cr", ascending=False).reset_index(drop=True)

    print(f"  Smart-money: {len(agg)} symbols across {len(deals)} deals "
          f"(window {deals['trade_date'].min().date()} → {deals['trade_date'].max().date()})")
    return agg, deals


if __name__ == "__main__":
    agg, deals = compute_smart_money()
    if not agg.empty:
        print(f"\n=== Net BUYING (smart money accumulating) ===")
        print(agg.head(12)[["symbol", "buy_value_cr", "sell_value_cr", "net_value_cr",
                            "n_unique_buyers", "repeat_accumulation"]].to_string(index=False))
        print(f"\n=== Net SELLING (distribution) ===")
        print(agg.tail(8)[["symbol", "buy_value_cr", "sell_value_cr", "net_value_cr"]].to_string(index=False))

        agg.to_parquet(DATA_DIR / "smart_money_signal.parquet", index=False)
        deals.to_parquet(DATA_DIR / "smart_money_deals.parquet", index=False)
        print(f"\nSaved {len(agg)} symbols, {len(deals)} deals")
