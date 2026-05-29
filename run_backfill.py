"""
Master backfill script.
Downloads equity + F&O bhavcopy for a date range, then loads into DuckDB.

Usage:
    python run_backfill.py --start 2024-01-01 --end 2024-12-31
    python run_backfill.py --days 30  # last 30 days
"""

import argparse
import sys
from datetime import date, timedelta

from src.ingestion.nse_equity_bhavcopy import backfill as equity_backfill
from src.ingestion.nse_fo_bhavcopy import backfill as fo_backfill
from src.ingestion.load_to_duckdb import load_all_available


def main():
    parser = argparse.ArgumentParser(description="Backfill NSE data and load to DuckDB")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=str(date.today()), help="End date YYYY-MM-DD")
    parser.add_argument("--days", type=int, help="Alternative: last N days (overrides --start)")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between requests (seconds)")
    parser.add_argument("--equity-only", action="store_true")
    parser.add_argument("--fo-only", action="store_true")
    parser.add_argument("--skip-load", action="store_true", help="Download only, don't load to DuckDB")
    args = parser.parse_args()

    end_date = date.fromisoformat(args.end)
    if args.days:
        start_date = end_date - timedelta(days=args.days)
    elif args.start:
        start_date = date.fromisoformat(args.start)
    else:
        start_date = end_date - timedelta(days=30)

    print(f"=== Backfill: {start_date} to {end_date} ===\n")

    if not args.fo_only:
        print("--- Equity Bhavcopy ---")
        equity_backfill(start_date, end_date, delay=args.delay)
        print()

    if not args.equity_only:
        print("--- F&O Bhavcopy ---")
        fo_backfill(start_date, end_date, delay=args.delay)
        print()

    if not args.skip_load:
        print("--- Loading to DuckDB ---")
        load_all_available()


if __name__ == "__main__":
    main()
