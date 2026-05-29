"""
NSE Corporate Filings Ingestion

Downloads corporate announcements, board meetings, corporate actions,
and promoter shareholding data from NSE APIs.

All endpoints require session cookies from nseindia.com homepage.
"""

import time
import json
import requests
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "corporate"

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def _get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass
    time.sleep(1)
    return session


def _nse_date(d: date) -> str:
    return d.strftime("%d-%m-%Y")


def _fetch_json(session: requests.Session, url: str, params: dict) -> Optional[list]:
    try:
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, list):
            return data
        return None
    except Exception as e:
        print(f"  Fetch error: {e}")
        return None


# --- Corporate Announcements ---

def download_announcements(
    start: date,
    end: date,
    session: Optional[requests.Session] = None,
    chunk_days: int = 30,
) -> Path:
    """Download corporate announcements in date-range chunks."""
    out_dir = RAW_DIR / "announcements"
    out_dir.mkdir(parents=True, exist_ok=True)

    if session is None:
        session = _get_session()

    all_records = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        params = {
            "index": "equities",
            "from_date": _nse_date(current),
            "to_date": _nse_date(chunk_end),
        }

        data = _fetch_json(
            session,
            "https://www.nseindia.com/api/corporate-announcements",
            params,
        )
        if data:
            all_records.extend(data)
            print(f"  Announcements {current} to {chunk_end}: {len(data)} records")
        else:
            print(f"  Announcements {current} to {chunk_end}: failed or empty")

        current = chunk_end + timedelta(days=1)
        time.sleep(1.5)

        if len(all_records) % 10000 < len(data or []):
            session = _get_session()

    out_path = out_dir / f"announcements_{start.isoformat()}_{end.isoformat()}.json"
    with open(out_path, "w") as f:
        json.dump(all_records, f)

    print(f"Total announcements: {len(all_records)} -> {out_path.name}")
    return out_path


# --- Board Meetings ---

def download_board_meetings(
    start: date,
    end: date,
    session: Optional[requests.Session] = None,
    chunk_days: int = 30,
) -> Path:
    out_dir = RAW_DIR / "board_meetings"
    out_dir.mkdir(parents=True, exist_ok=True)

    if session is None:
        session = _get_session()

    all_records = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        params = {
            "index": "equities",
            "from_date": _nse_date(current),
            "to_date": _nse_date(chunk_end),
        }

        data = _fetch_json(
            session,
            "https://www.nseindia.com/api/corporate-board-meetings",
            params,
        )
        if data:
            all_records.extend(data)
            print(f"  Board meetings {current} to {chunk_end}: {len(data)} records")
        else:
            print(f"  Board meetings {current} to {chunk_end}: failed or empty")

        current = chunk_end + timedelta(days=1)
        time.sleep(1.5)

    out_path = out_dir / f"board_meetings_{start.isoformat()}_{end.isoformat()}.json"
    with open(out_path, "w") as f:
        json.dump(all_records, f)

    print(f"Total board meetings: {len(all_records)} -> {out_path.name}")
    return out_path


# --- Corporate Actions ---

def download_corporate_actions(
    start: date,
    end: date,
    session: Optional[requests.Session] = None,
    chunk_days: int = 90,
) -> Path:
    out_dir = RAW_DIR / "corporate_actions"
    out_dir.mkdir(parents=True, exist_ok=True)

    if session is None:
        session = _get_session()

    all_records = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        params = {
            "index": "equities",
            "from_date": _nse_date(current),
            "to_date": _nse_date(chunk_end),
        }

        data = _fetch_json(
            session,
            "https://www.nseindia.com/api/corporates-corporateActions",
            params,
        )
        if data:
            all_records.extend(data)
            print(f"  Corp actions {current} to {chunk_end}: {len(data)} records")
        else:
            print(f"  Corp actions {current} to {chunk_end}: failed or empty")

        current = chunk_end + timedelta(days=1)
        time.sleep(1.5)

    out_path = out_dir / f"corporate_actions_{start.isoformat()}_{end.isoformat()}.json"
    with open(out_path, "w") as f:
        json.dump(all_records, f)

    print(f"Total corporate actions: {len(all_records)} -> {out_path.name}")
    return out_path


# --- Promoter Shareholding ---

def download_shareholding(
    symbols: list,
    session: Optional[requests.Session] = None,
) -> Path:
    out_dir = RAW_DIR / "shareholding"
    out_dir.mkdir(parents=True, exist_ok=True)

    if session is None:
        session = _get_session()

    all_records = []
    for i, symbol in enumerate(symbols):
        data = _fetch_json(
            session,
            "https://www.nseindia.com/api/corporate-share-holdings-master",
            {"index": "equities", "symbol": symbol},
        )
        if data:
            for rec in data:
                rec["_symbol"] = symbol
            all_records.extend(data)

        if (i + 1) % 50 == 0:
            print(f"  Shareholding: {i + 1}/{len(symbols)} symbols processed")
            session = _get_session()
        else:
            time.sleep(1)

    out_path = out_dir / f"shareholding_{date.today().isoformat()}.json"
    with open(out_path, "w") as f:
        json.dump(all_records, f)

    print(f"Total shareholding records: {len(all_records)} for {len(symbols)} symbols -> {out_path.name}")
    return out_path


# --- BSE Insider Trading ---

def download_insider_trading(
    start: date,
    end: date,
    session: Optional[requests.Session] = None,
    chunk_days: int = 30,
) -> Path:
    out_dir = RAW_DIR / "insider_trading"
    out_dir.mkdir(parents=True, exist_ok=True)

    if session is None:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "*/*",
            "Referer": "https://www.bseindia.com/",
        })
        try:
            session.get("https://www.bseindia.com/", timeout=10)
        except Exception:
            pass
        time.sleep(1)

    all_records = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)

        page = 1
        chunk_records = []
        while True:
            params = {
                "pageno": str(page),
                "strCat": "Insider Trading / SAST",
                "subcategory": "-1",
                "strPrevDate": current.strftime("%Y%m%d"),
                "strToDate": chunk_end.strftime("%Y%m%d"),
                "strType": "C",
                "strSearch": "",
                "strscrip": "",
            }

            try:
                resp = session.get(
                    "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w",
                    params=params,
                    timeout=30,
                )
                if resp.status_code != 200:
                    print(f"  BSE insider {current}-{chunk_end} page {page}: HTTP {resp.status_code}")
                    break

                data = resp.json()
                table = data.get("Table", [])
                if not table:
                    break

                chunk_records.extend(table)

                total = int(data.get("Table1", [{}])[0].get("ROWCNT", 0))
                if len(chunk_records) >= total:
                    break

                page += 1
                time.sleep(1)

            except Exception as e:
                print(f"  BSE insider error: {e}")
                break

        all_records.extend(chunk_records)
        print(f"  Insider trading {current} to {chunk_end}: {len(chunk_records)} records")

        current = chunk_end + timedelta(days=1)
        time.sleep(1.5)

    out_path = out_dir / f"insider_trading_{start.isoformat()}_{end.isoformat()}.json"
    with open(out_path, "w") as f:
        json.dump(all_records, f)

    print(f"Total insider trading: {len(all_records)} -> {out_path.name}")
    return out_path


def backfill_all(start: date, end: date):
    """Download all Tier 2 corporate data for a date range."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Corporate Announcements ===")
    download_announcements(start, end)

    print("\n=== Board Meetings ===")
    download_board_meetings(start, end)

    print("\n=== Corporate Actions ===")
    download_corporate_actions(start, end)

    print("\n=== BSE Insider Trading ===")
    download_insider_trading(start, end)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download NSE/BSE Corporate Filings")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=str(date.today()), help="End date YYYY-MM-DD")
    parser.add_argument("--type", choices=["all", "announcements", "board", "actions", "insider", "shareholding"],
                        default="all")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start) if args.start else date.today() - timedelta(days=90)
    end_date = date.fromisoformat(args.end)

    if args.type == "all":
        backfill_all(start_date, end_date)
    elif args.type == "announcements":
        download_announcements(start_date, end_date)
    elif args.type == "board":
        download_board_meetings(start_date, end_date)
    elif args.type == "actions":
        download_corporate_actions(start_date, end_date)
    elif args.type == "insider":
        download_insider_trading(start_date, end_date)
    elif args.type == "shareholding":
        print("Shareholding requires a symbol list. Use download_shareholding() directly.")
