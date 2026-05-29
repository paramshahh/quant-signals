"""
NSE FII Derivatives Statistics Ingestion

Daily FII/FPI buy/sell/OI data across index futures, index options,
stock futures, and stock options. Published as XLS files on archives.nseindia.com.

URL pattern: archives.nseindia.com/content/fo/fii_stats_{DD}-{Mmm}-{YYYY}.xls
"""

import time
import xlrd
import requests
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "fii_stats"

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}

CATEGORY_ROWS = {
    "INDEX FUTURES", "INDEX OPTIONS", "STOCK FUTURES", "STOCK OPTIONS",
    "BANKNIFTY FUTURES", "FINNIFTY FUTURES", "MIDCPNIFTY FUTURES",
    "NIFTY FUTURES", "NIFTYNXT50 FUTURES",
    "BANKNIFTY OPTIONS", "FINNIFTY OPTIONS", "MIDCPNIFTY OPTIONS",
    "NIFTY OPTIONS", "NIFTYNXT50 OPTIONS",
}


def _get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=5)
    except Exception:
        pass
    time.sleep(0.5)
    return session


def _fii_stats_url(d: date) -> str:
    date_str = d.strftime("%d-%b-%Y")
    return f"https://archives.nseindia.com/content/fo/fii_stats_{date_str}.xls"


def download_fii_stats(
    d: date,
    session: Optional[requests.Session] = None,
    force: bool = False,
) -> Optional[Path]:
    raw_path = RAW_DIR / f"{d.isoformat()}_fii_stats.xls"
    if raw_path.exists() and not force:
        return raw_path

    if session is None:
        session = _get_session()

    url = _fii_stats_url(d)
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[{d}] FII stats download failed: {e}")
        return None

    if len(resp.content) < 500:
        return None

    raw_path.write_bytes(resp.content)
    return raw_path


def parse_fii_stats(xls_path: Path) -> pd.DataFrame:
    wb = xlrd.open_workbook(str(xls_path))
    ws = wb.sheet_by_index(0)

    rows = []
    for i in range(ws.nrows):
        label = str(ws.cell_value(i, 0)).strip().upper()
        if label in {c.upper() for c in CATEGORY_ROWS}:
            def _num(v):
                try:
                    return float(str(v).replace(",", "").strip())
                except (ValueError, TypeError):
                    return None

            rows.append({
                "category": ws.cell_value(i, 0).strip(),
                "buy_contracts": _num(ws.cell_value(i, 1)),
                "buy_value_cr": _num(ws.cell_value(i, 2)),
                "sell_contracts": _num(ws.cell_value(i, 3)),
                "sell_value_cr": _num(ws.cell_value(i, 4)),
                "oi_contracts": _num(ws.cell_value(i, 5)),
                "oi_value_cr": _num(ws.cell_value(i, 6)),
            })

    return pd.DataFrame(rows)


def backfill(start: date, end: date, delay: float = 2.0):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = _get_session()

    current = start
    downloaded = 0
    skipped = 0

    while current <= end:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        result = download_fii_stats(current, session=session)
        if result:
            downloaded += 1
            print(f"[{current}] FII stats OK -> {result.name}")
        else:
            skipped += 1

        current += timedelta(days=1)
        time.sleep(delay)

        if (downloaded + skipped) % 40 == 0:
            session = _get_session()

    print(f"\nDone. Downloaded: {downloaded}, Skipped/holidays: {skipped}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download NSE FII Derivatives Statistics")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=str(date.today()), help="End date YYYY-MM-DD")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between requests")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start) if args.start else date.today() - timedelta(days=30)
    end_date = date.fromisoformat(args.end)

    print(f"Backfilling FII stats: {start_date} to {end_date}")
    backfill(start_date, end_date, delay=args.delay)
