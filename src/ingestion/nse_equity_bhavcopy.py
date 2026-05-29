"""
NSE Equity Bhavcopy Ingestion

Downloads daily equity bhavcopy files from NSE archives.
Contains OHLCV + delivery quantity + delivery percentage for all listed scrips.

Two formats:
- Legacy (pre July 8, 2024): CSV zip at /content/historical/EQUITIES/{year}/{mon}/cm{ddMMMyyyy}bhav.csv.zip
- UDiFF (post July 8, 2024): CSV at /api/reports?archives=[{"name":"CM - Bhavcopy(Full)","type":"archives","category":"capital-market","section":"equities"}]&date=DD-MM-YYYY

We target the legacy format for historical backfill and UDiFF for recent/daily.
"""

import os
import time
import hashlib
import requests
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "equity_bhavcopy"

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

UDIFF_CUTOVER = date(2024, 7, 8)


def _get_session() -> requests.Session:
    """Create a session. archives.nseindia.com doesn't need cookie auth."""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=5)
    except Exception:
        pass
    time.sleep(0.5)
    return session


def _file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def _legacy_url(d: date) -> str:
    """Pre-July 2024 format: sec_bhavdata_full includes delivery qty and pct."""
    date_str = d.strftime("%d%m%Y")
    return f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"


def _udiff_url(d: date) -> str:
    """Post-July 2024 UDiFF format via reports API."""
    date_str = d.strftime("%d-%m-%Y")
    return (
        f"https://www.nseindia.com/api/reports?archives="
        f'[{{"name":"CM - Bhavcopy(Full)","type":"archives","category":"capital-market","section":"equities"}}]'
        f"&date={date_str}&type=equities&mode=single"
    )


def download_equity_bhavcopy(
    d: date,
    session: Optional[requests.Session] = None,
    force: bool = False,
) -> Optional[Path]:
    """
    Download equity bhavcopy for a given date.
    Returns path to saved raw CSV, or None if market was closed / download failed.
    """
    raw_path = RAW_DIR / f"{d.isoformat()}_equity_bhavcopy.csv"
    if raw_path.exists() and not force:
        return raw_path

    if session is None:
        session = _get_session()

    # sec_bhavdata_full works for all dates (legacy and post-UDiFF cutover)
    url = _legacy_url(d)

    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[{d}] Download failed: {e}")
        return None

    csv_bytes = resp.content

    if len(csv_bytes) < 100:
        return None

    file_hash = _file_hash(csv_bytes)
    meta_path = RAW_DIR / f"{d.isoformat()}_equity_bhavcopy.meta"

    raw_path.write_bytes(csv_bytes)
    meta_path.write_text(
        f"source={url}\n"
        f"download_ts={datetime.utcnow().isoformat()}Z\n"
        f"sha256_prefix={file_hash}\n"
        f"size_bytes={len(csv_bytes)}\n"
    )

    return raw_path


def parse_equity_bhavcopy(csv_path: Path) -> pd.DataFrame:
    """
    Parse a raw equity bhavcopy CSV into a clean DataFrame.
    Normalizes column names across legacy and UDiFF formats.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.upper()

    # Legacy format columns: SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, LAST, PREVCLOSE,
    #                         TOTTRDQTY, TOTTRDVAL, TIMESTAMP, TOTALTRADES, ISIN, DELIVQTY, DELVRPRCNT
    # UDiFF format may differ slightly — normalize here

    col_map = {
        "TOTTRDQTY": "VOLUME",
        "TOTTRDVAL": "TURNOVER",
        "DELIVQTY": "DELIVERY_QTY",
        "DELVRPRCNT": "DELIVERY_PCT",
        "TTL_TRD_QNTY": "VOLUME",
        "TURNOVER_LACS": "TURNOVER",
        "DLVRY_QTY": "DELIVERY_QTY",
        "DELIV_QTY": "DELIVERY_QTY",
        "DELIV_PER": "DELIVERY_PCT",
        "NO_OF_TRADES": "TRADES",
        "OPEN_PRICE": "OPEN",
        "HIGH_PRICE": "HIGH",
        "LOW_PRICE": "LOW",
        "CLOSE_PRICE": "CLOSE",
        "PREV_CLOSE": "PREV_CLOSE",
    }

    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Filter to EQ series (main board equity)
    if "SERIES" in df.columns:
        df = df[df["SERIES"].isin(["EQ", "BE", "BZ", "SM", "ST"])].copy()

    keep_cols = [
        "SYMBOL", "SERIES", "OPEN", "HIGH", "LOW", "CLOSE",
        "VOLUME", "TURNOVER", "DELIVERY_QTY", "DELIVERY_PCT",
    ]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    for col in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "TURNOVER", "DELIVERY_QTY", "DELIVERY_PCT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def backfill(start: date, end: date, delay: float = 1.5):
    """Download equity bhavcopy for a date range with respectful delays."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = _get_session()

    current = start
    downloaded = 0
    skipped = 0

    while current <= end:
        if current.weekday() >= 5:  # skip weekends
            current += timedelta(days=1)
            continue

        result = download_equity_bhavcopy(current, session=session)
        if result:
            downloaded += 1
            print(f"[{current}] OK -> {result.name}")
        else:
            skipped += 1

        current += timedelta(days=1)
        time.sleep(delay)

        # Refresh session every 50 requests to avoid cookie expiry
        if (downloaded + skipped) % 50 == 0:
            session = _get_session()

    print(f"\nDone. Downloaded: {downloaded}, Skipped/holidays: {skipped}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download NSE Equity Bhavcopy")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=str(date.today()), help="End date YYYY-MM-DD")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between requests in seconds")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start) if args.start else date.today() - timedelta(days=30)
    end_date = date.fromisoformat(args.end)

    print(f"Backfilling equity bhavcopy: {start_date} to {end_date}")
    backfill(start_date, end_date, delay=args.delay)
