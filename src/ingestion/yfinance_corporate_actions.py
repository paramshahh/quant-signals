"""
yfinance corporate-action seed for the deep historical TRI (pre-2023 only).

The legacy CM bhavcopy gives deep prices but NO corporate actions, so a 2015-2022 momentum
panel would re-acquire phantom split/bonus crashes. yfinance carries clean unadjusted
split + dividend history for NSE tickers going back decades. We pull it, synthesize subject
strings the existing TRI parser already understands, and write to PARQUET (no DB write — runs
concurrently with the price-seed's DuckDB write lock).

Only ex-dates BEFORE 2023-01-02 are kept: that's where we have no NSE corporate_actions, so
there's no double-adjust risk against the live data.

yfinance "Stock Splits" value r is the share multiplier (1->r), so the backward price factor
is 1/r — encoded as a split subject "From Rs {r} To Rs 1" (TRI parser -> CAF = 1/r). This
captures both splits and bonus issues (yfinance reports a 1:1 bonus as a 2.0 split).

Output: data/raw/yf_corporate_actions.parquet  (symbol, ex_date, subject, source)
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parents[2]
UNIVERSE = BASE / "data" / "_universe.txt"
OUT = BASE / "data" / "raw" / "yf_corporate_actions.parquet"
CUTOFF = pd.Timestamp("2023-01-02")     # only seed where we lack NSE CA

# NSE symbols whose yfinance ticker needs the & encoded etc. (rare); default is SYMBOL.NS
def _yf_ticker(sym: str) -> str:
    return f"{sym}.NS"


def fetch_symbol(sym: str):
    rows = []
    try:
        act = yf.Ticker(_yf_ticker(sym)).actions
    except Exception:
        return rows
    if act is None or act.empty:
        return rows
    if getattr(act.index, "tz", None) is not None:
        act.index = act.index.tz_localize(None)     # yfinance returns tz-aware index
    act = act[act.index < CUTOFF]
    for ts, r in act.iterrows():
        ex = pd.Timestamp(ts).date()
        split = float(r.get("Stock Splits", 0) or 0)
        div = float(r.get("Dividends", 0) or 0)
        if split and abs(split - 1.0) > 1e-9:
            # share multiplier r -> backward price factor 1/r; encode as a split subject
            rows.append({"symbol": sym, "ex_date": ex,
                         "subject": f"Stock Split (yfinance) - From Rs {split:g} Per Share To Rs 1 Per Share",
                         "source": "yfinance"})
        if div and div > 0:
            rows.append({"symbol": sym, "ex_date": ex,
                         "subject": f"Dividend (yfinance) - Rs {div:g} Per Share",
                         "source": "yfinance"})
    return rows


def run(delay: float = 0.3):
    symbols = [s.strip() for s in UNIVERSE.read_text().splitlines() if s.strip()]
    print(f"Fetching yfinance corporate actions for {len(symbols)} symbols (ex-date < {CUTOFF.date()})")
    all_rows, ok, empty, err = [], 0, 0, 0
    for i, sym in enumerate(symbols):
        try:
            rows = fetch_symbol(sym)
            if rows:
                all_rows.extend(rows); ok += 1
            else:
                empty += 1
        except Exception:
            err += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(symbols)}] actions={len(all_rows)} ok={ok} empty={empty} err={err}", flush=True)
            if all_rows:
                pd.DataFrame(all_rows).to_parquet(OUT, index=False)   # incremental checkpoint
        time.sleep(delay)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["ex_date"] = pd.to_datetime(df["ex_date"])
        df.to_parquet(OUT, index=False)
    n_split = int((df["subject"].str.contains("Split")).sum()) if not df.empty else 0
    n_div = int((df["subject"].str.contains("Dividend")).sum()) if not df.empty else 0
    print(f"\nDone: {len(df)} pre-2023 actions ({n_split} split/bonus, {n_div} dividend) "
          f"for {ok} symbols -> {OUT.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--test", type=str, help="Test one symbol")
    args = ap.parse_args()
    if args.test:
        print(fetch_symbol(args.test.upper()))
    else:
        run(args.delay)
