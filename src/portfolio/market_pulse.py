"""
Market Pulse — the "how is the market doing / thinking" time series for the trading cockpit.

Two outputs:
  1. MARKET BREADTH (historical, from prices) — % of the liquid universe above its 200-DMA /
     50-DMA and % with positive 12-1 momentum, weekly over the full panel. This is the classic
     risk-on/risk-off gauge and it's fully reconstructable from adjusted_prices (so it shows the
     2018 liquidity crash and 2020 COVID breadth collapse immediately).
  2. CONVICTION HISTORY (append-only, forward) — each run logs market avg/median conviction,
     % high-conviction, and YOUR BOOK's avg conviction. Conviction isn't cheaply reconstructable
     historically (it depends on snapshot fundamentals), so this series starts today and accrues.

Outputs: data/export/market_breadth.json, data/export/conviction_history.json
         (+ append-only data/paper/conviction_history.parquet)
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DB_PATH = BASE / "data" / "quant_signals.duckdb"
EXPORT = BASE / "data" / "export"
HIST = BASE / "data" / "paper" / "conviction_history.parquet"

# mirror the portfolio_tracker book (NSE symbols)
MY_HOLDINGS = {"MUTHOOTFIN","MCX","APOLLOHOSP","SOLARINDS","CAPLIPOINT","KARURVYSYA","APTUS",
               "AFFLE","HSCL","CHOLAFIN","GVT&D","ANANTRAJ","CUB","CENTRALBK","TORNTPHARM"}

MIN_ADTV_LAKH = 100.0     # liquid universe for breadth


def market_breadth(con) -> pd.DataFrame:
    px = con.execute("""
        SELECT trade_date, canonical_symbol AS symbol, adj_close, turnover
        FROM adjusted_prices WHERE adj_close > 0
    """).fetchdf()
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    close = px.pivot_table(index="trade_date", columns="symbol", values="adj_close").sort_index()
    turn = px.pivot_table(index="trade_date", columns="symbol", values="turnover").reindex_like(close)

    sma200 = close.rolling(200, min_periods=100).mean()
    sma50 = close.rolling(50, min_periods=30).mean()
    mom = close.shift(21) / close.shift(252) - 1.0
    adtv = turn.rolling(20, min_periods=10).mean()
    liquid = adtv >= MIN_ADTV_LAKH

    def pct(mask_true, valid):
        m = mask_true & valid
        denom = valid.sum(axis=1)
        return (m.sum(axis=1) / denom.replace(0, np.nan)) * 100

    out = pd.DataFrame({
        "date": close.index,
        "pct_above_200dma": pct(close > sma200, liquid & sma200.notna()).values,
        "pct_above_50dma": pct(close > sma50, liquid & sma50.notna()).values,
        "pct_pos_momentum": pct(mom > 0, liquid & mom.notna()).values,
    }).dropna()
    # weekly downsample (every 5th trading day) keeps the chart light over 11 years
    out = out.iloc[::5].reset_index(drop=True)
    return out


def log_conviction_snapshot(con):
    cs = pd.read_parquet(BASE / "data" / "conviction_score.parquet")
    cs = cs[cs["final_score"].notna()]
    today = pd.to_datetime(con.execute("SELECT MAX(trade_date) FROM adjusted_prices").fetchone()[0])
    held = cs[cs["symbol"].isin(MY_HOLDINGS)]
    row = {
        "date": today,
        "mkt_avg": float(cs["conviction"].mean()),
        "mkt_median": float(cs["conviction"].median()),
        "pct_high": float((cs["conviction"] >= 80).mean() * 100),
        "my_avg": float(held["conviction"].mean()) if len(held) else np.nan,
        "n_scored": int(len(cs)),
    }
    hist = pd.read_parquet(HIST) if HIST.exists() else pd.DataFrame()
    if not hist.empty:
        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist[hist["date"] != today]      # idempotent: replace today's row
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True).sort_values("date")
    hist.to_parquet(HIST, index=False)
    return hist


def _to_json(df, path):
    import json
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype(str)
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(2)
    recs = df.to_dict("records")
    for r in recs:
        for k, v in r.items():
            if isinstance(v, float) and v != v:
                r[k] = None
    json.dump(recs, open(path, "w"))
    return len(recs)


def run(con=None):
    should_close = con is None
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)
    EXPORT.mkdir(parents=True, exist_ok=True)

    breadth = market_breadth(con)
    n1 = _to_json(breadth, EXPORT / "market_breadth.json")
    hist = log_conviction_snapshot(con)
    n2 = _to_json(hist, EXPORT / "conviction_history.json")
    if should_close:
        con.close()

    latest = breadth.iloc[-1]
    h = hist.iloc[-1]
    print(f"  market_breadth: {n1} weekly points | latest {latest['date'].date()}: "
          f"{latest['pct_above_200dma']:.0f}% >200DMA, {latest['pct_pos_momentum']:.0f}% +mom")
    print(f"  conviction_history: {n2} day(s) | today mkt_avg {h['mkt_avg']:.1f}, my_book {h['my_avg']:.1f}")
    return breadth, hist


if __name__ == "__main__":
    run()
