"""
NSE Bulk & Block Deals Ingestion

Bulk deals: trades >= 0.5% of listed equity shares.
Block deals: trades >= 5 lakh shares or Rs.5 crore in the block window.

NSE only exposes the current day's file at a static URL (no historical archives).
We download daily and store with a date prefix for accumulation over time.
"""

import time
import requests
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "bulk_block"

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}

BULK_URL = "https://archives.nseindia.com/content/equities/bulk.csv"
BLOCK_URL = "https://archives.nseindia.com/content/equities/block.csv"


def _get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=5)
    except Exception:
        pass
    time.sleep(0.5)
    return session


def download_bulk_deals(
    session: Optional[requests.Session] = None,
    force: bool = False,
) -> Optional[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if session is None:
        session = _get_session()

    try:
        resp = session.get(BULK_URL, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Bulk deals download failed: {e}")
        return None

    content = resp.text.strip()
    if len(content) < 50 or "NO RECORDS" in content:
        return None

    lines = content.split("\n")
    if len(lines) < 2:
        return None

    first_data = lines[1].split(",")[0].strip()
    try:
        file_date = datetime.strptime(first_data, "%d-%b-%Y").date()
    except ValueError:
        file_date = date.today()

    raw_path = RAW_DIR / f"{file_date.isoformat()}_bulk_deals.csv"
    if raw_path.exists() and not force:
        return raw_path

    raw_path.write_text(content)
    return raw_path


def download_block_deals(
    session: Optional[requests.Session] = None,
    force: bool = False,
) -> Optional[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if session is None:
        session = _get_session()

    try:
        resp = session.get(BLOCK_URL, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Block deals download failed: {e}")
        return None

    content = resp.text.strip()
    if len(content) < 50 or "NO RECORDS" in content:
        return None

    lines = content.split("\n")
    if len(lines) < 2:
        return None

    first_data = lines[1].split(",")[0].strip()
    try:
        file_date = datetime.strptime(first_data, "%d-%b-%Y").date()
    except ValueError:
        file_date = date.today()

    raw_path = RAW_DIR / f"{file_date.isoformat()}_block_deals.csv"
    if raw_path.exists() and not force:
        return raw_path

    raw_path.write_text(content)
    return raw_path


def parse_bulk_deals(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    col_map = {
        "Date": "TRADE_DATE",
        "Symbol": "SYMBOL",
        "Security Name": "SECURITY_NAME",
        "Client Name": "CLIENT_NAME",
        "Buy/Sell": "BUY_SELL",
        "Quantity Traded": "QUANTITY",
        "Trade Price / Wght. Avg. Price": "PRICE",
        "Remarks": "REMARKS",
    }
    df.rename(columns=col_map, inplace=True)

    if "TRADE_DATE" in df.columns:
        df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"], format="%d-%b-%Y", errors="coerce").dt.date

    for col in ["QUANTITY"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["PRICE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def parse_block_deals(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    col_map = {
        "Date": "TRADE_DATE",
        "Symbol": "SYMBOL",
        "Security Name": "SECURITY_NAME",
        "Client Name": "CLIENT_NAME",
        "Buy/Sell": "BUY_SELL",
        "Quantity Traded": "QUANTITY",
        "Trade Price / Wght. Avg. Price": "PRICE",
    }
    df.rename(columns=col_map, inplace=True)

    if "TRADE_DATE" in df.columns:
        df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"], format="%d-%b-%Y", errors="coerce").dt.date

    for col in ["QUANTITY"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["PRICE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def download_today(force: bool = False):
    """Download today's bulk and block deals."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = _get_session()

    bulk = download_bulk_deals(session=session, force=force)
    if bulk:
        print(f"Bulk deals -> {bulk.name}")
    else:
        print("No bulk deals today (or download failed)")

    block = download_block_deals(session=session, force=force)
    if block:
        print(f"Block deals -> {block.name}")
    else:
        print("No block deals today (or download failed)")


if __name__ == "__main__":
    download_today()
