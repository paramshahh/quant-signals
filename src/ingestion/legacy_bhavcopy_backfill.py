"""
Legacy CM bhavcopy backfill — one-time STATIC seed of deep price history (2008–2022).

Architecture (per the multi-regime panel plan): history is a frozen one-time seed, the daily
cron is left untouched on the current sec_bhavdata_full format. This module is the seed job.

The legacy NSE CM bhavcopy (cmDDMONYYYYbhav.csv.zip) carries OHLC + volume + turnover + ISIN
(no delivery %, which momentum/SUE don't need) and is available back to ~2008. It is parsed
into the existing equity_daily schema (delivery columns NULL) and INSERT-ed only for dates not
already present, so it composes cleanly with the live 2023+ data without overwriting it.

Bonus: the legacy file has ISIN, so we also seed a `symbol_isin` master that strengthens the
Trap-4 rename bridge historically.

Usage:
  python -m src.ingestion.legacy_bhavcopy_backfill --start 2016-01-01 --end 2022-12-31
  python -m src.ingestion.legacy_bhavcopy_backfill --test 2020-03-23   # parse one day, no write
"""

import argparse
import io
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import requests

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "quant_signals.duckdb"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nseindia.com/",
}
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _url(d: date) -> str:
    mon = MONTHS[d.month - 1]
    return (f"https://archives.nseindia.com/content/historical/EQUITIES/"
            f"{d.year}/{mon}/cm{d.day:02d}{mon}{d.year}bhav.csv.zip")


def download_legacy_day(d: date, session: requests.Session):
    """Return a normalized DataFrame matching equity_daily, or None (holiday/404/error)."""
    try:
        r = session.get(_url(d), timeout=20)
    except requests.RequestException:
        return None
    if r.status_code != 200 or len(r.content) < 2000:
        return None
    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open(z.namelist()[0]))
    except Exception:
        return None

    df.columns = [c.strip() for c in df.columns]
    df = df[df["SERIES"].astype(str).str.strip() == "EQ"].copy()
    if df.empty:
        return None

    out = pd.DataFrame({
        "trade_date": d,
        "symbol": df["SYMBOL"].astype(str).str.strip(),
        "series": "EQ",
        "open": pd.to_numeric(df["OPEN"], errors="coerce"),
        "high": pd.to_numeric(df["HIGH"], errors="coerce"),
        "low": pd.to_numeric(df["LOW"], errors="coerce"),
        "close": pd.to_numeric(df["CLOSE"], errors="coerce"),
        "volume": pd.to_numeric(df["TOTTRDQTY"], errors="coerce").astype("Int64"),
        "turnover": pd.to_numeric(df["TOTTRDVAL"], errors="coerce") / 1e5,  # rupees -> lakh (match live)
        "delivery_qty": pd.Series([pd.NA] * len(df), dtype="Int64"),
        "delivery_pct": pd.Series([float("nan")] * len(df), dtype="float64"),
        "isin": df["ISIN"].astype(str).str.strip(),
    })
    return out[out["close"] > 0]


def backfill(start: date, end: date, delay: float = 0.4):
    con = duckdb.connect(str(DB_PATH))
    existing = set(con.execute(
        "SELECT DISTINCT trade_date FROM equity_daily WHERE series='EQ'"
    ).fetchdf()["trade_date"].astype(str))
    con.execute("CREATE TABLE IF NOT EXISTS symbol_isin (symbol VARCHAR, isin VARCHAR, last_seen DATE)")

    session = requests.Session()
    session.headers.update(HEADERS)

    d, loaded, skipped, holidays, isin_rows = start, 0, 0, 0, []
    while d <= end:
        if d.weekday() >= 5:                       # weekend
            d += timedelta(days=1); continue
        if d.isoformat() in existing:              # already have this day (live data)
            skipped += 1; d += timedelta(days=1); continue

        df = download_legacy_day(d, session)
        if df is None:
            holidays += 1
        else:
            isin_rows.append(df[["symbol", "isin"]].assign(last_seen=d))
            con.register("day_df", df.drop(columns=["isin"]))
            con.execute("INSERT INTO equity_daily SELECT * FROM day_df")
            con.unregister("day_df")
            loaded += 1
            if loaded % 50 == 0:
                print(f"  [{d}] loaded={loaded} skipped={skipped} holidays={holidays}", flush=True)
        time.sleep(delay)
        d += timedelta(days=1)

    if isin_rows:
        allisin = pd.concat(isin_rows, ignore_index=True)
        allisin = allisin[allisin["isin"].str.startswith("INE", na=False)]
        con.register("isin_df", allisin)
        con.execute("INSERT INTO symbol_isin SELECT symbol, isin, last_seen FROM isin_df")
        con.unregister("isin_df")

    print(f"\nDone: {loaded} days loaded, {skipped} already-present skipped, {holidays} holidays/404")
    print(f"equity_daily now: {con.execute('SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM equity_daily').fetchone()}")
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str)
    ap.add_argument("--end", type=str)
    ap.add_argument("--delay", type=float, default=0.4)
    ap.add_argument("--test", type=str, help="Parse a single date (YYYY-MM-DD), no DB write")
    args = ap.parse_args()

    if args.test:
        s = requests.Session(); s.headers.update(HEADERS)
        df = download_legacy_day(date.fromisoformat(args.test), s)
        if df is None:
            print("No data (holiday / 404)")
        else:
            print(f"Parsed {len(df)} EQ rows for {args.test}")
            print(df[["symbol", "open", "high", "low", "close", "volume", "turnover", "isin"]].head(5).to_string(index=False))
            print(f"\nSchema matches equity_daily (delivery cols NULL). ISIN present: {df['isin'].str.startswith('INE').mean()*100:.0f}%")
    else:
        backfill(date.fromisoformat(args.start), date.fromisoformat(args.end), args.delay)
