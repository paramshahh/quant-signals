"""
Daily data refresh pipeline for quant-signals.

Run after market close (~6pm IST) on trading days.
Downloads today's data, loads into DuckDB, recomputes signals, exports for dashboard.

Usage:
    python daily_refresh.py              # full pipeline
    python daily_refresh.py --fetch-only # download only, no signal recompute
    python daily_refresh.py --signals-only # recompute signals + export (no download)
"""

import sys
import time
import subprocess
from datetime import date, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "data" / "logs"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_step(description, cmd):
    log(f"START: {description}")
    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=str(PROJECT_DIR), capture_output=True, text=True, timeout=300
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f"FAIL: {description} ({elapsed:.1f}s)")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-5:]:
                log(f"  stderr: {line}")
        return False

    last_lines = [l for l in result.stdout.strip().split("\n") if l][-3:]
    for line in last_lines:
        log(f"  {line}")
    log(f"OK: {description} ({elapsed:.1f}s)")
    return True


def fetch_daily_data():
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    steps = [
        (
            "Equity bhavcopy",
            [sys.executable, "-m", "src.ingestion.nse_equity_bhavcopy",
             "--start", yesterday, "--end", today, "--delay", "1.0"],
        ),
        (
            "F&O bhavcopy + participant OI/vol",
            [sys.executable, "-m", "src.ingestion.nse_fo_bhavcopy",
             "--start", yesterday, "--end", today, "--delay", "1.5"],
        ),
        (
            "FII derivatives stats",
            [sys.executable, "-m", "src.ingestion.nse_fii_stats",
             "--start", yesterday, "--end", today, "--delay", "1.0"],
        ),
        (
            "Bulk & block deals",
            [sys.executable, "-m", "src.ingestion.nse_bulk_block"],
        ),
    ]

    success = 0
    for desc, cmd in steps:
        if run_step(desc, cmd):
            success += 1
        time.sleep(1)

    return success, len(steps)


def load_to_db():
    return run_step(
        "Load raw files into DuckDB",
        [sys.executable, "-m", "src.ingestion.load_to_duckdb"],
    )


def compute_signals():
    steps = [
        (
            "Total Return Index (corporate-action price adjustment)",
            [sys.executable, "-m", "src.processing.total_return_index"],
        ),
        (
            "Signal 1: CADP (pre-earnings delivery)",
            [sys.executable, "-m", "src.signals.cadp"],
        ),
        (
            "Signal 2: F&O positioning",
            [sys.executable, "-m", "src.signals.fo_positioning"],
        ),
        (
            "Signal 3: Corporate Health Index",
            [sys.executable, "-m", "src.signals.chi"],
        ),
        (
            "Signal 4: Macro nowcast",
            [sys.executable, "-m", "src.signals.macro_nowcast"],
        ),
        (
            "Signal 5: SUE (earnings surprise)",
            [sys.executable, "-m", "src.signals.sue"],
        ),
        (
            "Signal 7: Price momentum",
            [sys.executable, "-m", "src.signals.momentum"],
        ),
        (
            "Signal 8: Value + Quality",
            [sys.executable, "-m", "src.signals.value_quality"],
        ),
        (
            "Signal 9: Smart-money (bulk/block)",
            [sys.executable, "-m", "src.signals.smart_money"],
        ),
        (
            "MasterScore (composite — depends on 1,2,3,5,7,8)",
            [sys.executable, "-m", "src.signals.master_score"],
        ),
        (
            "Conviction Score (unified — blocks + shrinkage + agreement)",
            [sys.executable, "-m", "src.signals.conviction_score"],
        ),
    ]

    success = 0
    for desc, cmd in steps:
        if run_step(desc, cmd):
            success += 1

    return success, len(steps)


def export_dashboard():
    return run_step(
        "Export data for dashboard",
        [sys.executable, "export_for_dashboard.py"],
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Daily quant-signals refresh")
    parser.add_argument("--fetch-only", action="store_true", help="Download only, skip signals")
    parser.add_argument("--signals-only", action="store_true", help="Recompute signals only, skip download")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log(f"DAILY REFRESH — {date.today().isoformat()}")
    log("=" * 60)

    t_start = time.time()

    if not args.signals_only:
        log("\n── PHASE 1: Fetch daily data ──")
        fetch_ok, fetch_total = fetch_daily_data()
        log(f"\nFetch: {fetch_ok}/{fetch_total} succeeded")

        log("\n── PHASE 2: Load into DuckDB ──")
        load_to_db()

    if not args.fetch_only:
        log("\n── PHASE 3: Compute signals ──")
        sig_ok, sig_total = compute_signals()
        log(f"\nSignals: {sig_ok}/{sig_total} succeeded")

        log("\n── PHASE 4: Export for dashboard ──")
        export_dashboard()

    elapsed = time.time() - t_start
    log(f"\n{'=' * 60}")
    log(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
