"""
Paper-trade tracker — the live out-of-sample track record.

A backtest is in-sample by construction; a PM knows it. What can't be manufactured later is a
dated, mark-to-market forward record. This freezes the top-decile Conviction book and marks it
daily vs Nifty 500 (^CRSLDX) — append-only, so once the clock starts it accrues forever.

Rules (per Param): top decile by Conviction (4+ blocks), EQUAL WEIGHT, MONTHLY rebalance.
Within a month the book is held; daily portfolio return = mean of held names' daily returns
(daily re-equal-weighted — the standard factor-research convention).

State (append-only, immutable history):
  data/paper/holdings.parquet : one row per (rebalance_date, symbol, entry_price, weight)
  data/paper/nav.parquet      : one row per date (port_value, bench_value, both =100 at inception)

Run daily (wired into daily_refresh): establishes the book on first run, marks it forward after.
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DB_PATH = BASE / "data" / "quant_signals.duckdb"
PAPER = BASE / "data" / "paper"
HOLDINGS = PAPER / "holdings.parquet"
NAV = PAPER / "nav.parquet"
EXPORT = BASE / "data" / "export"

TOP_DECILE = 0.10
MIN_BLOCKS = 4
REBALANCE_DAYS = 25     # ~monthly


def _bench_close(dates):
    """Nifty 500 (^CRSLDX) close on/just-before each date, via yfinance; None if unavailable."""
    try:
        import yfinance as yf
        h = yf.Ticker("^CRSLDX").history(period="5y", auto_adjust=False)
        if h.empty:
            return None
        s = h["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s
    except Exception:
        return None


def run(con=None):
    should_close = con is None
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)

    cs = pd.read_parquet(BASE / "data" / "conviction_score.parquet")
    cs = cs[(cs["final_score"].notna()) & (cs["n_blocks"] >= MIN_BLOCKS)].copy()
    px = con.execute("""
        SELECT trade_date, canonical_symbol AS symbol, adj_close
        FROM adjusted_prices WHERE adj_close > 0
    """).fetchdf()
    if should_close:
        con.close()
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    today = px["trade_date"].max()
    close_today = px[px["trade_date"] == today].set_index("symbol")["adj_close"]

    holdings = pd.read_parquet(HOLDINGS) if HOLDINGS.exists() else pd.DataFrame()
    nav = pd.read_parquet(NAV) if NAV.exists() else pd.DataFrame()
    if not holdings.empty:
        holdings["rebalance_date"] = pd.to_datetime(holdings["rebalance_date"])
    if not nav.empty:
        nav["date"] = pd.to_datetime(nav["date"])

    # ── rebalance? (first run, or >= REBALANCE_DAYS since last rebalance) ──
    last_reb = holdings["rebalance_date"].max() if not holdings.empty else None
    due = holdings.empty or (today - last_reb).days >= REBALANCE_DAYS
    if due:
        cutoff = cs["conviction"].quantile(1 - TOP_DECILE)
        picks = cs[cs["conviction"] >= cutoff].copy()
        picks = picks[picks["symbol"].isin(close_today.index)]
        w = 1.0 / len(picks)
        new = pd.DataFrame({
            "rebalance_date": today, "symbol": picks["symbol"].values,
            "entry_price": close_today.reindex(picks["symbol"]).values,
            "weight": w, "conviction": picks["conviction"].values,
        })
        holdings = pd.concat([holdings, new], ignore_index=True)
        holdings.to_parquet(HOLDINGS, index=False)
        print(f"  REBALANCE {today.date()}: {len(new)} names, equal-weight {w*100:.2f}% each")

    active = holdings[holdings["rebalance_date"] == holdings["rebalance_date"].max()]

    # ── mark today's NAV (idempotent) ──
    if nav.empty or today not in set(nav["date"]):
        bench = _bench_close(None)
        if nav.empty:
            port_val, bench_val = 100.0, 100.0          # inception
            port_ret = bench_ret = 0.0
        else:
            prev = nav.sort_values("date").iloc[-1]
            prev_date = prev["date"]
            prev_close = px[px["trade_date"] == prev_date].set_index("symbol")["adj_close"]
            r = (close_today.reindex(active["symbol"]) / prev_close.reindex(active["symbol"]) - 1.0)
            port_ret = float(r.dropna().mean()) if r.notna().any() else 0.0
            port_val = prev["port_value"] * (1 + port_ret)
            bench_ret = 0.0
            if bench is not None and prev_date in bench.index and today in bench.index:
                bench_ret = float(bench.loc[today] / bench.loc[prev_date] - 1.0)
            bench_val = prev["bench_value"] * (1 + bench_ret)
        row = pd.DataFrame([{"date": today, "port_value": port_val, "bench_value": bench_val,
                             "port_ret": port_ret, "bench_ret": bench_ret, "n_holdings": len(active)}])
        nav = pd.concat([nav, row], ignore_index=True)
        nav.to_parquet(NAV, index=False)

    # ── export current book + nav for the dashboard ──
    EXPORT.mkdir(parents=True, exist_ok=True)
    book = active.merge(cs[["symbol", "conviction", "b_earnings", "b_value", "b_quality",
                            "b_momentum", "b_flow"]], on="symbol", how="left", suffixes=("", "_cs"))
    book = book.sort_values("conviction", ascending=False)
    _to_json(book, EXPORT / "paper_book.json")
    _to_json(nav.sort_values("date"), EXPORT / "paper_nav.json")

    nv = nav.sort_values("date").iloc[-1]
    incept = nav.sort_values("date").iloc[0]["date"]
    print(f"  Paper book: {len(active)} names | inception {pd.Timestamp(incept).date()} | "
          f"NAV {nv['port_value']:.2f} vs bench {nv['bench_value']:.2f} "
          f"(+{nv['port_value']-100:.2f} vs +{nv['bench_value']-100:.2f})")
    return holdings, nav


def _to_json(df, path):
    import json
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype(str)
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(4)
    recs = df.to_dict("records")
    for r in recs:
        for k, v in r.items():
            if isinstance(v, float) and v != v:
                r[k] = None
    json.dump(recs, open(path, "w"))


if __name__ == "__main__":
    run()
