"""
Load raw bhavcopy CSVs into DuckDB for fast analytical queries.
Maintains point-in-time correctness: each row carries the trade_date it belongs to.
"""

import duckdb
import pandas as pd
from datetime import date
from pathlib import Path
from typing import Optional

from .nse_equity_bhavcopy import parse_equity_bhavcopy, RAW_DIR as EQUITY_RAW_DIR
from .nse_fo_bhavcopy import parse_fo_bhavcopy, parse_participant_oi, RAW_DIR as FO_RAW_DIR
from .nse_bulk_block import parse_bulk_deals, parse_block_deals, RAW_DIR as BULK_BLOCK_RAW_DIR
from .nse_fii_stats import parse_fii_stats, RAW_DIR as FII_RAW_DIR
from .macro_gst import download_and_parse_all as gst_download_and_parse, RAW_DIR as GST_RAW_DIR
from .macro_rbi_credit import download_and_parse_all as rbi_download_and_parse, parse_rbi_credit_excel, RAW_DIR as RBI_CREDIT_RAW_DIR

CORPORATE_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "corporate"
MACRO_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "macro"

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "quant_signals.duckdb"


def get_db() -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    _init_schema(con)
    return con


def _init_schema(con: duckdb.DuckDBPyConnection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS equity_daily (
            trade_date DATE,
            symbol VARCHAR,
            series VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            turnover DOUBLE,
            delivery_qty BIGINT,
            delivery_pct DOUBLE,
            PRIMARY KEY (trade_date, symbol, series)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS fo_daily (
            trade_date DATE,
            symbol VARCHAR,
            instrument VARCHAR,
            expiry DATE,
            strike DOUBLE,
            option_type VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            settle_pr DOUBLE,
            contracts BIGINT,
            value_lakhs DOUBLE,
            oi BIGINT,
            oi_change BIGINT
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS participant_oi (
            trade_date DATE,
            client_type VARCHAR,
            future_index_long BIGINT,
            future_index_short BIGINT,
            future_stock_long BIGINT,
            future_stock_short BIGINT,
            option_index_call_long BIGINT,
            option_index_put_long BIGINT,
            option_index_call_short BIGINT,
            option_index_put_short BIGINT,
            option_stock_call_long BIGINT,
            option_stock_put_long BIGINT,
            option_stock_call_short BIGINT,
            option_stock_put_short BIGINT,
            total_long BIGINT,
            total_short BIGINT,
            PRIMARY KEY (trade_date, client_type)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS participant_vol (
            trade_date DATE,
            client_type VARCHAR,
            future_index_long BIGINT,
            future_index_short BIGINT,
            future_stock_long BIGINT,
            future_stock_short BIGINT,
            option_index_call_long BIGINT,
            option_index_put_long BIGINT,
            option_index_call_short BIGINT,
            option_index_put_short BIGINT,
            option_stock_call_long BIGINT,
            option_stock_put_long BIGINT,
            option_stock_call_short BIGINT,
            option_stock_put_short BIGINT,
            total_long BIGINT,
            total_short BIGINT,
            PRIMARY KEY (trade_date, client_type)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS bulk_deals (
            trade_date DATE,
            symbol VARCHAR,
            security_name VARCHAR,
            client_name VARCHAR,
            buy_sell VARCHAR,
            quantity BIGINT,
            price DOUBLE
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS block_deals (
            trade_date DATE,
            symbol VARCHAR,
            security_name VARCHAR,
            client_name VARCHAR,
            buy_sell VARCHAR,
            quantity BIGINT,
            price DOUBLE
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS fii_stats (
            trade_date DATE,
            category VARCHAR,
            buy_contracts BIGINT,
            buy_value_cr DOUBLE,
            sell_contracts BIGINT,
            sell_value_cr DOUBLE,
            oi_contracts BIGINT,
            oi_value_cr DOUBLE,
            PRIMARY KEY (trade_date, category)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS corporate_announcements (
            announcement_date TIMESTAMP,
            symbol VARCHAR,
            company_name VARCHAR,
            industry VARCHAR,
            category VARCHAR,
            subject TEXT,
            attachment_url VARCHAR,
            isin VARCHAR,
            seq_id VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS board_meetings (
            meeting_date DATE,
            symbol VARCHAR,
            company_name VARCHAR,
            industry VARCHAR,
            purpose TEXT,
            description TEXT,
            meeting_type VARCHAR,
            intimation_type VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS corporate_actions (
            symbol VARCHAR,
            company_name VARCHAR,
            series VARCHAR,
            face_value VARCHAR,
            subject TEXT,
            ex_date DATE,
            record_date DATE,
            isin VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS promoter_shareholding (
            symbol VARCHAR,
            company_name VARCHAR,
            quarter_end DATE,
            promoter_pct DOUBLE,
            public_pct DOUBLE,
            employee_trusts_pct DOUBLE,
            submission_date DATE
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS gst_collections (
            month DATE,
            state_code INTEGER,
            state VARCHAR,
            cgst_cr DOUBLE,
            sgst_cr DOUBLE,
            igst_cr DOUBLE,
            total_cr DOUBLE,
            PRIMARY KEY (month, state_code)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS rbi_sectoral_credit (
            report_date DATE,
            statement VARCHAR,
            sector VARCHAR,
            depth INTEGER,
            outstanding_cr DOUBLE,
            yoy_pct DOUBLE
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS insider_trading (
            security_code INTEGER,
            security_name VARCHAR,
            person_name VARCHAR,
            category VARCHAR,
            pre_securities_type VARCHAR,
            pre_securities_qty BIGINT,
            pre_securities_pct DOUBLE,
            acquired_securities_type VARCHAR,
            acquired_qty BIGINT,
            acquired_value DOUBLE,
            transaction_type VARCHAR,
            post_securities_type VARCHAR,
            post_securities_qty BIGINT,
            post_pct DOUBLE,
            date_from DATE,
            date_to DATE,
            intimation_date DATE,
            mode_of_acquisition VARCHAR,
            derivative_type VARCHAR,
            derivative_spec VARCHAR,
            derivative_buy_value DOUBLE,
            derivative_buy_units BIGINT,
            derivative_sell_value DOUBLE,
            derivative_sell_units BIGINT,
            exchange VARCHAR,
            reported_date DATE
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS iip_index (
            period DATE,
            sector VARCHAR,
            index_value DOUBLE,
            PRIMARY KEY (period, sector)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_log (
            source VARCHAR,
            trade_date DATE,
            loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            row_count INTEGER,
            PRIMARY KEY (source, trade_date)
        )
    """)


def _already_loaded(con: duckdb.DuckDBPyConnection, source: str, trade_date: date) -> bool:
    result = con.execute(
        "SELECT 1 FROM ingestion_log WHERE source = ? AND trade_date = ?",
        [source, trade_date]
    ).fetchone()
    return result is not None


def load_equity_day(con: duckdb.DuckDBPyConnection, trade_date: date, force: bool = False) -> int:
    if not force and _already_loaded(con, "equity", trade_date):
        return 0

    csv_path = EQUITY_RAW_DIR / f"{trade_date.isoformat()}_equity_bhavcopy.csv"
    if not csv_path.exists():
        return 0

    df = parse_equity_bhavcopy(csv_path)
    if df.empty:
        return 0

    df["trade_date"] = trade_date

    col_map = {
        "SYMBOL": "symbol",
        "SERIES": "series",
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "VOLUME": "volume",
        "TURNOVER": "turnover",
        "DELIVERY_QTY": "delivery_qty",
        "DELIVERY_PCT": "delivery_pct",
    }
    df.rename(columns=col_map, inplace=True)

    keep = ["trade_date"] + [v for v in col_map.values() if v in df.columns]
    df = df[[c for c in keep if c in df.columns]]

    if force:
        con.execute("DELETE FROM equity_daily WHERE trade_date = ?", [trade_date])

    con.execute("INSERT OR IGNORE INTO equity_daily SELECT * FROM df")
    con.execute(
        "INSERT OR REPLACE INTO ingestion_log (source, trade_date, row_count) VALUES (?, ?, ?)",
        ["equity", trade_date, len(df)]
    )
    return len(df)


def load_fo_day(con: duckdb.DuckDBPyConnection, trade_date: date, force: bool = False) -> int:
    if not force and _already_loaded(con, "fo", trade_date):
        return 0

    csv_path = FO_RAW_DIR / f"{trade_date.isoformat()}_fo_bhavcopy.csv"
    if not csv_path.exists():
        return 0

    df = parse_fo_bhavcopy(csv_path)
    if df.empty:
        return 0

    df["trade_date"] = trade_date

    col_map = {
        "SYMBOL": "symbol",
        "INSTRUMENT": "instrument",
        "EXPIRY": "expiry",
        "STRIKE": "strike",
        "OPTION_TYPE": "option_type",
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "SETTLE_PR": "settle_pr",
        "CONTRACTS": "contracts",
        "VALUE_LAKHS": "value_lakhs",
        "OI": "oi",
        "OI_CHANGE": "oi_change",
    }
    df.rename(columns=col_map, inplace=True)

    keep = ["trade_date"] + [v for v in col_map.values() if v in df.columns]
    df = df[[c for c in keep if c in df.columns]]

    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date

    if force:
        con.execute("DELETE FROM fo_daily WHERE trade_date = ?", [trade_date])

    con.execute("INSERT INTO fo_daily SELECT * FROM df")
    con.execute(
        "INSERT OR REPLACE INTO ingestion_log (source, trade_date, row_count) VALUES (?, ?, ?)",
        ["fo", trade_date, len(df)]
    )
    return len(df)


def _parse_participant_file(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skiprows=1)
    df.columns = df.columns.str.strip()

    col_map = {
        "Client Type": "client_type",
        "Future Index Long": "future_index_long",
        "Future Index Short": "future_index_short",
        "Future Stock Long": "future_stock_long",
        "Future Stock Short": "future_stock_short",
        "Option Index Call Long": "option_index_call_long",
        "Option Index Put Long": "option_index_put_long",
        "Option Index Call Short": "option_index_call_short",
        "Option Index Put Short": "option_index_put_short",
        "Option Stock Call Long": "option_stock_call_long",
        "Option Stock Put Long": "option_stock_put_long",
        "Option Stock Call Short": "option_stock_call_short",
        "Option Stock Put Short": "option_stock_put_short",
        "Total Long Contracts": "total_long",
        "Total Short Contracts": "total_short",
    }
    df.rename(columns=col_map, inplace=True)

    keep = [v for v in col_map.values() if v in df.columns]
    df = df[keep]
    df = df[df["client_type"].isin(["Client", "DII", "FII", "Pro"])]

    for col in df.columns:
        if col != "client_type":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_participant_oi_day(con: duckdb.DuckDBPyConnection, trade_date: date, force: bool = False) -> int:
    if not force and _already_loaded(con, "participant_oi", trade_date):
        return 0

    csv_path = FO_RAW_DIR / f"{trade_date.isoformat()}_participant_oi.csv"
    if not csv_path.exists():
        return 0

    df = _parse_participant_file(csv_path)
    if df.empty:
        return 0

    df.insert(0, "trade_date", trade_date)

    if force:
        con.execute("DELETE FROM participant_oi WHERE trade_date = ?", [trade_date])

    con.execute("INSERT OR IGNORE INTO participant_oi SELECT * FROM df")
    con.execute(
        "INSERT OR REPLACE INTO ingestion_log (source, trade_date, row_count) VALUES (?, ?, ?)",
        ["participant_oi", trade_date, len(df)]
    )
    return len(df)


def load_participant_vol_day(con: duckdb.DuckDBPyConnection, trade_date: date, force: bool = False) -> int:
    if not force and _already_loaded(con, "participant_vol", trade_date):
        return 0

    csv_path = FO_RAW_DIR / f"{trade_date.isoformat()}_participant_vol.csv"
    if not csv_path.exists():
        return 0

    df = _parse_participant_file(csv_path)
    if df.empty:
        return 0

    df.insert(0, "trade_date", trade_date)

    if force:
        con.execute("DELETE FROM participant_vol WHERE trade_date = ?", [trade_date])

    con.execute("INSERT OR IGNORE INTO participant_vol SELECT * FROM df")
    con.execute(
        "INSERT OR REPLACE INTO ingestion_log (source, trade_date, row_count) VALUES (?, ?, ?)",
        ["participant_vol", trade_date, len(df)]
    )
    return len(df)


def load_bulk_deals_file(con: duckdb.DuckDBPyConnection, csv_path: Path, force: bool = False) -> int:
    trade_date = date.fromisoformat(csv_path.name[:10])

    if not force and _already_loaded(con, "bulk_deals", trade_date):
        return 0

    df = parse_bulk_deals(csv_path)
    if df.empty:
        return 0

    out = pd.DataFrame({
        "trade_date": trade_date,
        "symbol": df.get("SYMBOL"),
        "security_name": df.get("SECURITY_NAME"),
        "client_name": df.get("CLIENT_NAME"),
        "buy_sell": df.get("BUY_SELL"),
        "quantity": df.get("QUANTITY"),
        "price": df.get("PRICE"),
    })

    if force:
        con.execute("DELETE FROM bulk_deals WHERE trade_date = ?", [trade_date])

    con.execute("INSERT INTO bulk_deals SELECT * FROM out")
    con.execute(
        "INSERT OR REPLACE INTO ingestion_log (source, trade_date, row_count) VALUES (?, ?, ?)",
        ["bulk_deals", trade_date, len(out)]
    )
    return len(out)


def load_block_deals_file(con: duckdb.DuckDBPyConnection, csv_path: Path, force: bool = False) -> int:
    trade_date = date.fromisoformat(csv_path.name[:10])

    if not force and _already_loaded(con, "block_deals", trade_date):
        return 0

    df = parse_block_deals(csv_path)
    if df.empty:
        return 0

    out = pd.DataFrame({
        "trade_date": trade_date,
        "symbol": df.get("SYMBOL"),
        "security_name": df.get("SECURITY_NAME"),
        "client_name": df.get("CLIENT_NAME"),
        "buy_sell": df.get("BUY_SELL"),
        "quantity": df.get("QUANTITY"),
        "price": df.get("PRICE"),
    })

    if force:
        con.execute("DELETE FROM block_deals WHERE trade_date = ?", [trade_date])

    con.execute("INSERT INTO block_deals SELECT * FROM out")
    con.execute(
        "INSERT OR REPLACE INTO ingestion_log (source, trade_date, row_count) VALUES (?, ?, ?)",
        ["block_deals", trade_date, len(out)]
    )
    return len(out)


def load_fii_stats_day(con: duckdb.DuckDBPyConnection, trade_date: date, force: bool = False) -> int:
    if not force and _already_loaded(con, "fii_stats", trade_date):
        return 0

    xls_path = FII_RAW_DIR / f"{trade_date.isoformat()}_fii_stats.xls"
    if not xls_path.exists():
        return 0

    df = parse_fii_stats(xls_path)
    if df.empty:
        return 0

    df.insert(0, "trade_date", trade_date)

    if force:
        con.execute("DELETE FROM fii_stats WHERE trade_date = ?", [trade_date])

    con.execute("INSERT OR IGNORE INTO fii_stats SELECT * FROM df")
    con.execute(
        "INSERT OR REPLACE INTO ingestion_log (source, trade_date, row_count) VALUES (?, ?, ?)",
        ["fii_stats", trade_date, len(df)]
    )
    return len(df)


def load_corporate_announcements(con: duckdb.DuckDBPyConnection, json_path: Path) -> int:
    import json as _json
    with open(json_path) as f:
        records = _json.load(f)
    if not records:
        return 0

    rows = []
    for r in records:
        sort_date = r.get("sort_date")
        rows.append({
            "announcement_date": sort_date,
            "symbol": r.get("symbol"),
            "company_name": r.get("sm_name"),
            "industry": r.get("smIndustry"),
            "category": r.get("desc"),
            "subject": r.get("attchmntText", "")[:2000] if r.get("attchmntText") else None,
            "attachment_url": r.get("attchmntFile"),
            "isin": r.get("sm_isin"),
            "seq_id": r.get("seq_id"),
        })

    df = pd.DataFrame(rows)
    df["announcement_date"] = pd.to_datetime(df["announcement_date"], errors="coerce")

    # Merge with existing rows and dedup (idempotent across multiple date-range files)
    existing = con.execute("SELECT * FROM corporate_announcements").fetchdf()
    combined = pd.concat([existing, df], ignore_index=True).drop_duplicates(
        subset=["seq_id"], keep="last")
    con.execute("DELETE FROM corporate_announcements")
    con.execute("INSERT INTO corporate_announcements SELECT * FROM combined")
    return len(df)


def load_board_meetings(con: duckdb.DuckDBPyConnection, json_path: Path) -> int:
    import json as _json
    with open(json_path) as f:
        records = _json.load(f)
    if not records:
        return 0

    rows = []
    for r in records:
        rows.append({
            "meeting_date": r.get("bm_date"),
            "symbol": r.get("bm_symbol"),
            "company_name": r.get("sm_name"),
            "industry": r.get("sm_indusrty"),
            "purpose": r.get("bm_purpose"),
            "description": r.get("bm_desc", "")[:2000] if r.get("bm_desc") else None,
            "meeting_type": r.get("meetingType"),
            "intimation_type": r.get("intimationType"),
        })

    df = pd.DataFrame(rows)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"], format="%d-%b-%Y", errors="coerce").dt.date

    existing = con.execute("SELECT * FROM board_meetings").fetchdf()
    combined = pd.concat([existing, df], ignore_index=True).drop_duplicates(
        subset=["symbol", "meeting_date", "purpose"], keep="last")
    con.execute("DELETE FROM board_meetings")
    con.execute("INSERT INTO board_meetings SELECT * FROM combined")
    return len(df)


def load_corporate_actions(con: duckdb.DuckDBPyConnection, json_path: Path) -> int:
    import json as _json
    with open(json_path) as f:
        records = _json.load(f)
    if not records:
        return 0

    rows = []
    for r in records:
        rows.append({
            "symbol": r.get("symbol"),
            "company_name": r.get("comp"),
            "series": r.get("series"),
            "face_value": r.get("faceVal"),
            "subject": r.get("subject"),
            "ex_date": r.get("exDate"),
            "record_date": r.get("recDate"),
            "isin": r.get("isin"),
        })

    df = pd.DataFrame(rows)
    for col in ["ex_date", "record_date"]:
        df[col] = pd.to_datetime(df[col], format="%d-%b-%Y", errors="coerce").dt.date

    existing = con.execute("SELECT * FROM corporate_actions").fetchdf()
    combined = pd.concat([existing, df], ignore_index=True).drop_duplicates(
        subset=["symbol", "ex_date", "subject"], keep="last")
    con.execute("DELETE FROM corporate_actions")
    con.execute("INSERT INTO corporate_actions SELECT * FROM combined")
    return len(df)


def load_promoter_shareholding(con: duckdb.DuckDBPyConnection, json_path: Path) -> int:
    import json as _json
    with open(json_path) as f:
        records = _json.load(f)
    if not records:
        return 0

    rows = []
    for r in records:
        rows.append({
            "symbol": r.get("_symbol") or r.get("symbol"),
            "company_name": r.get("name"),
            "quarter_end": r.get("date"),
            "promoter_pct": r.get("pr_and_prgrp"),
            "public_pct": r.get("public_val"),
            "employee_trusts_pct": r.get("employeeTrusts"),
            "submission_date": r.get("submissionDate"),
        })

    df = pd.DataFrame(rows)
    df["quarter_end"] = pd.to_datetime(df["quarter_end"], format="%d-%b-%Y", errors="coerce").dt.date
    df["submission_date"] = pd.to_datetime(df["submission_date"], errors="coerce").dt.date
    for col in ["promoter_pct", "public_pct", "employee_trusts_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    con.execute("DELETE FROM promoter_shareholding")
    con.execute("INSERT INTO promoter_shareholding SELECT * FROM df")
    return len(df)


def load_gst_collections(con: duckdb.DuckDBPyConnection) -> int:
    """Load GST collections from downloaded Excel files."""
    from .macro_gst import download_and_parse_all
    df = download_and_parse_all()
    if df.empty:
        return 0
    con.execute("DELETE FROM gst_collections")
    con.execute("INSERT INTO gst_collections SELECT * FROM df")
    return len(df)


def load_rbi_sectoral_credit(con: duckdb.DuckDBPyConnection) -> int:
    """Load RBI sectoral credit from downloaded Excel files."""
    files = sorted(RBI_CREDIT_RAW_DIR.glob("rbi_credit_*.xlsx")) if RBI_CREDIT_RAW_DIR.exists() else []
    if not files:
        return 0

    all_dfs = []
    for f in files:
        report_date = date.fromisoformat(f.stem.replace("rbi_credit_", ""))
        df = parse_rbi_credit_excel(f, report_date)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return 0

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["report_date"] = pd.to_datetime(combined["report_date"])
    con.execute("DELETE FROM rbi_sectoral_credit")
    con.execute("INSERT INTO rbi_sectoral_credit SELECT * FROM combined")
    return len(combined)


def load_insider_trading(con: duckdb.DuckDBPyConnection) -> int:
    """Load BSE insider trading from manually downloaded CSVs."""
    insider_dir = CORPORATE_RAW_DIR / "insider_trading"
    if not insider_dir.exists():
        return 0

    files = sorted(insider_dir.glob("*.csv"))
    if not files:
        return 0

    all_dfs = []
    for f in files:
        if f.name.startswith("_"):
            continue
        try:
            df = pd.read_csv(f)
            all_dfs.append(df)
        except Exception:
            continue

    if not all_dfs:
        return 0

    combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates()

    col_map = {
        "Security Code": "security_code",
        "Security Name": "security_name",
        "Name of Person": "person_name",
        "Category of person": "category",
        "Type of Securities held Prior to acquisition/Disposed)": "pre_securities_type",
        "Number of Securities held Prior to acquisition/Disposed": "pre_securities_qty",
        "%   of  Securities held Prior to acquisition/Disposed": "pre_securities_pct",
        "Type of Securities Acquired/Disposed/Pledge etc.": "acquired_securities_type",
        "Number of Securities Acquired/Disposed/Pledge etc.": "acquired_qty",
        "Value  of Securities Acquired/Disposed/Pledge etc": "acquired_value",
        "Transaction Type ( Buy/Sale/Pledge/Revoke/Invoke)": "transaction_type",
        "Type of Securities held Post  acquisition/Disposed/Pledge  etc": "post_securities_type",
        "Number of Securities held Post  acquisition/Disposed/Pledge etc": "post_securities_qty",
        "Post-Transaction % of Shareholding": "post_pct",
        "Date of acquisition of shares/sale of shares/Date of Allotment(From date)": "date_from",
        "Date of acquisition of shares/sale of shares/Date of Allotment( To date  )": "date_to",
        "Date of Intimation to Company": "intimation_date",
        "Mode of Acquisition": "mode_of_acquisition",
        "Trading in Derivative - Type of Contract": "derivative_type",
        "Derivative Contract  Specification": "derivative_spec",
        "Derivatives - Buy Value (Notional)": "derivative_buy_value",
        "Derivatives - Buy-Number of Units(Contracts*lot size)": "derivative_buy_units",
        "Derivatives - Sales Value (Notional)": "derivative_sell_value",
        "Derivatives - Sales-Number of Units(Contracts * lot size)": "derivative_sell_units",
        "Exchange on which the Trade was executed": "exchange",
        "Reported to Exchange": "reported_date",
    }
    combined.rename(columns=col_map, inplace=True)
    keep = [v for v in col_map.values() if v in combined.columns]
    combined = combined[keep]

    for date_col in ["date_from", "date_to", "intimation_date", "reported_date"]:
        if date_col in combined.columns:
            combined[date_col] = pd.to_datetime(combined[date_col], format="%d %b %Y", errors="coerce").dt.date

    for num_col in ["security_code", "pre_securities_qty", "acquired_qty", "post_securities_qty",
                     "derivative_buy_units", "derivative_sell_units"]:
        if num_col in combined.columns:
            combined[num_col] = pd.to_numeric(combined[num_col], errors="coerce")

    for float_col in ["pre_securities_pct", "acquired_value", "post_pct",
                       "derivative_buy_value", "derivative_sell_value"]:
        if float_col in combined.columns:
            combined[float_col] = pd.to_numeric(combined[float_col], errors="coerce")

    con.execute("DELETE FROM insider_trading")
    con.execute("INSERT INTO insider_trading SELECT * FROM combined")
    return len(combined)


def load_iip_index(con: duckdb.DuckDBPyConnection) -> int:
    """Load IIP data from manually downloaded files."""
    from .macro_iip import parse_all_iip_files
    df = parse_all_iip_files()
    if df.empty:
        return 0
    con.execute("DELETE FROM iip_index")
    con.execute("INSERT INTO iip_index SELECT * FROM df")
    return len(df)


def load_all_available(force: bool = False):
    """Scan raw directories and load any unloaded days into DuckDB."""
    con = get_db()

    equity_files = sorted(EQUITY_RAW_DIR.glob("*_equity_bhavcopy.csv"))
    fo_files = sorted(FO_RAW_DIR.glob("*_fo_bhavcopy.csv"))
    participant_oi_files = sorted(FO_RAW_DIR.glob("*_participant_oi.csv"))
    participant_vol_files = sorted(FO_RAW_DIR.glob("*_participant_vol.csv"))
    bulk_files = sorted(BULK_BLOCK_RAW_DIR.glob("*_bulk_deals.csv")) if BULK_BLOCK_RAW_DIR.exists() else []
    block_files = sorted(BULK_BLOCK_RAW_DIR.glob("*_block_deals.csv")) if BULK_BLOCK_RAW_DIR.exists() else []
    fii_files = sorted(FII_RAW_DIR.glob("*_fii_stats.xls")) if FII_RAW_DIR.exists() else []

    eq_loaded = 0
    for f in equity_files:
        trade_date = date.fromisoformat(f.name[:10])
        rows = load_equity_day(con, trade_date, force=force)
        if rows:
            eq_loaded += 1
            print(f"  Equity {trade_date}: {rows} rows")

    fo_loaded = 0
    for f in fo_files:
        trade_date = date.fromisoformat(f.name[:10])
        rows = load_fo_day(con, trade_date, force=force)
        if rows:
            fo_loaded += 1
            print(f"  F&O {trade_date}: {rows} rows")

    poi_loaded = 0
    for f in participant_oi_files:
        trade_date = date.fromisoformat(f.name[:10])
        rows = load_participant_oi_day(con, trade_date, force=force)
        if rows:
            poi_loaded += 1

    pvol_loaded = 0
    for f in participant_vol_files:
        trade_date = date.fromisoformat(f.name[:10])
        rows = load_participant_vol_day(con, trade_date, force=force)
        if rows:
            pvol_loaded += 1

    bulk_loaded = 0
    for f in bulk_files:
        rows = load_bulk_deals_file(con, f, force=force)
        if rows:
            bulk_loaded += 1

    block_loaded = 0
    for f in block_files:
        rows = load_block_deals_file(con, f, force=force)
        if rows:
            block_loaded += 1

    fii_loaded = 0
    for f in fii_files:
        trade_date = date.fromisoformat(f.name[:10])
        rows = load_fii_stats_day(con, trade_date, force=force)
        if rows:
            fii_loaded += 1

    # --- Tier 2: Corporate filings (merge across all downloaded date-range files) ---
    print("\n--- Tier 2: Corporate Filings ---")
    corp_base = CORPORATE_RAW_DIR
    for sub, loader in [
        ("announcements", load_corporate_announcements),
        ("board_meetings", load_board_meetings),
        ("corporate_actions", load_corporate_actions),
    ]:
        d = corp_base / sub
        if not d.exists():
            continue
        for jf in sorted(d.glob("*.json")):
            try:
                n = loader(con, jf)
                if n:
                    print(f"  {sub}: +{n} from {jf.name}")
            except Exception as e:
                print(f"  {sub}: FAILED on {jf.name}: {e}")

    # --- Tier 3: Macro data ---
    print("\n--- Tier 3: Macro Data ---")

    gst_rows = load_gst_collections(con)
    if gst_rows:
        print(f"  GST collections: {gst_rows} records")

    rbi_rows = load_rbi_sectoral_credit(con)
    if rbi_rows:
        print(f"  RBI sectoral credit: {rbi_rows} records")

    insider_rows = load_insider_trading(con)
    if insider_rows:
        print(f"  Insider trading: {insider_rows} records")

    iip_rows = 0
    try:
        iip_rows = load_iip_index(con)
        if iip_rows:
            print(f"  IIP index: {iip_rows} records")
    except Exception as e:
        print(f"  IIP index: SKIPPED (loader error, non-fatal): {e}")

    print(f"\nLoaded into {DB_PATH}:")
    print(f"  Equity: {eq_loaded} days")
    print(f"  F&O: {fo_loaded} days")
    print(f"  Participant OI: {poi_loaded} days")
    print(f"  Participant Vol: {pvol_loaded} days")
    print(f"  Bulk deals: {bulk_loaded} days")
    print(f"  Block deals: {block_loaded} days")
    print(f"  FII stats: {fii_loaded} days")
    print(f"  GST collections: {gst_rows} records")
    print(f"  RBI sectoral credit: {rbi_rows} records")
    print(f"  Insider trading: {insider_rows} records")
    print(f"  IIP index: {iip_rows} records")
    con.close()


if __name__ == "__main__":
    load_all_available()
