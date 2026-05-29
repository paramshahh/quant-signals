"""
NSE F&O Bhavcopy Ingestion

Downloads daily F&O bhavcopy files from NSE archives.
Contains contract-level OI, change in OI, volumes, settlement prices for all futures & options.

Two formats:
- Legacy (pre July 8, 2024): fo{ddMMMyyyy}bhav.csv.zip from /content/historical/DERIVATIVES/
- UDiFF (post July 8, 2024): "F&O - Bhavcopy(Full)" via reports API

Also downloads participant-wise OI and trading volumes (FII/DII/Pro/Client breakdown).
"""

import os
import io
import time
import zipfile
import hashlib
import requests
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "fo_bhavcopy"

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

UDIFF_CUTOVER = date(2024, 7, 8)


def _get_session() -> requests.Session:
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
    """Pre-July 2024: fo01JAN2024bhav.csv.zip from historical archives."""
    mon_str = d.strftime("%b").upper()
    return (
        f"https://archives.nseindia.com/content/historical/DERIVATIVES/"
        f"{d.year}/{mon_str}/fo{d.strftime('%d%b%Y').upper()}bhav.csv.zip"
    )


def _udiff_fo_url(d: date) -> str:
    """Post-July 2024: UDiFF format zip from /content/fo/."""
    date_str = d.strftime("%Y%m%d")
    return f"https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date_str}_F_0000.csv.zip"


def _udiff_url(d: date) -> str:
    """Post-July 2024 UDiFF format."""
    date_str = d.strftime("%d-%m-%Y")
    return (
        f"https://www.nseindia.com/api/reports?archives="
        f'[{{"name":"F&O - Bhavcopy(Full)","type":"archives","category":"derivatives","section":"equity"}}]'
        f"&date={date_str}&type=equity&mode=single"
    )


def _participant_oi_url(d: date) -> str:
    """Participant-wise open interest (FII/DII/Pro/Client)."""
    date_str = d.strftime("%d%m%Y")
    return f"https://archives.nseindia.com/content/nsccl/fao_participant_oi_{date_str}.csv"


def _participant_vol_url(d: date) -> str:
    """Participant-wise trading volumes."""
    date_str = d.strftime("%d%m%Y")
    return f"https://archives.nseindia.com/content/nsccl/fao_participant_vol_{date_str}.csv"


def download_fo_bhavcopy(
    d: date,
    session: Optional[requests.Session] = None,
    force: bool = False,
) -> Optional[Path]:
    """Download F&O bhavcopy for a given date."""
    raw_path = RAW_DIR / f"{d.isoformat()}_fo_bhavcopy.csv"
    if raw_path.exists() and not force:
        return raw_path

    if session is None:
        session = _get_session()

    if d >= UDIFF_CUTOVER:
        url = _udiff_fo_url(d)
    else:
        url = _legacy_url(d)

    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[{d}] F&O download failed: {e}")
        return None

    try:
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = z.namelist()[0]
        csv_bytes = z.read(csv_name)
    except zipfile.BadZipFile:
        print(f"[{d}] Bad zip file")
        return None

    if len(csv_bytes) < 100:
        return None

    file_hash = _file_hash(csv_bytes)
    meta_path = RAW_DIR / f"{d.isoformat()}_fo_bhavcopy.meta"

    raw_path.write_bytes(csv_bytes)
    meta_path.write_text(
        f"source={url}\n"
        f"download_ts={datetime.utcnow().isoformat()}Z\n"
        f"sha256_prefix={file_hash}\n"
        f"size_bytes={len(csv_bytes)}\n"
    )
    return raw_path


def download_participant_data(
    d: date,
    session: Optional[requests.Session] = None,
    force: bool = False,
) -> dict:
    """Download participant-wise OI and volume files."""
    results = {}
    if session is None:
        session = _get_session()

    for label, url_fn in [("participant_oi", _participant_oi_url), ("participant_vol", _participant_vol_url)]:
        raw_path = RAW_DIR / f"{d.isoformat()}_{label}.csv"
        if raw_path.exists() and not force:
            results[label] = raw_path
            continue

        url = url_fn(d)
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            if len(resp.content) < 50:
                continue
            raw_path.write_bytes(resp.content)
            results[label] = raw_path
        except requests.RequestException:
            continue

    return results


def parse_fo_bhavcopy(csv_path: Path) -> pd.DataFrame:
    """
    Parse F&O bhavcopy into clean DataFrame.
    Handles both legacy format (pre Jul 2024) and UDiFF format (post Jul 2024).
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.upper()

    # Detect format by checking for UDiFF-specific columns
    is_udiff = "TCKRSYMB" in df.columns or "FININSTRMTP" in df.columns

    if is_udiff:
        col_map = {
            "TCKRSYMB": "SYMBOL",
            "FININSTRMTP": "INSTRUMENT",
            "XPRYDT": "EXPIRY",
            "STRKPRIC": "STRIKE",
            "OPTNTP": "OPTION_TYPE",
            "OPNPRIC": "OPEN",
            "HGHPRIC": "HIGH",
            "LWPRIC": "LOW",
            "CLSPRIC": "CLOSE",
            "STTLMPRIC": "SETTLE_PR",
            "TTLTRADGVOL": "CONTRACTS",
            "TTLTRFVAL": "VALUE_LAKHS",
            "OPNINTRST": "OI",
            "CHNGИНOPNINTRST": "OI_CHANGE",
            "CHNGИНOPNINTRST": "OI_CHANGE",
            "CHNGINOPNINTRST": "OI_CHANGE",
        }
        # UDiFF instrument types: STO (stock options), STF (stock futures),
        # IDO (index options), IDF (index futures)
        instrument_map = {
            "STO": "OPTSTK",
            "STF": "FUTSTK",
            "IDO": "OPTIDX",
            "IDF": "FUTIDX",
        }
    else:
        col_map = {
            "EXPIRY_DT": "EXPIRY",
            "STRIKE_PR": "STRIKE",
            "OPTION_TYP": "OPTION_TYPE",
            "VAL_INLAKH": "VALUE_LAKHS",
            "OPEN_INT": "OI",
            "CHG_IN_OI": "OI_CHANGE",
        }
        instrument_map = None

    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    if instrument_map and "INSTRUMENT" in df.columns:
        df["INSTRUMENT"] = df["INSTRUMENT"].map(instrument_map).fillna(df["INSTRUMENT"])

    numeric_cols = ["OPEN", "HIGH", "LOW", "CLOSE", "SETTLE_PR", "STRIKE",
                    "CONTRACTS", "VALUE_LAKHS", "OI", "OI_CHANGE"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def parse_participant_oi(csv_path: Path) -> pd.DataFrame:
    """Parse participant-wise OI file (FII/DII/Pro/Client breakdown)."""
    df = pd.read_csv(csv_path, skiprows=1)
    df.columns = df.columns.str.strip()
    return df


def backfill(start: date, end: date, delay: float = 2.0, include_participants: bool = True):
    """Download F&O bhavcopy + participant data for a date range."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = _get_session()

    current = start
    downloaded = 0
    skipped = 0

    while current <= end:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        result = download_fo_bhavcopy(current, session=session)
        if result:
            downloaded += 1
            print(f"[{current}] F&O OK -> {result.name}")

            if include_participants:
                p_results = download_participant_data(current, session=session)
                if p_results:
                    print(f"  + participant files: {list(p_results.keys())}")
        else:
            skipped += 1

        current += timedelta(days=1)
        time.sleep(delay)

        if (downloaded + skipped) % 40 == 0:
            session = _get_session()

    print(f"\nDone. Downloaded: {downloaded}, Skipped/holidays: {skipped}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download NSE F&O Bhavcopy")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=str(date.today()), help="End date YYYY-MM-DD")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between requests")
    parser.add_argument("--no-participants", action="store_true", help="Skip participant OI/vol files")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start) if args.start else date.today() - timedelta(days=30)
    end_date = date.fromisoformat(args.end)

    print(f"Backfilling F&O bhavcopy: {start_date} to {end_date}")
    backfill(start_date, end_date, delay=args.delay, include_participants=not args.no_participants)
