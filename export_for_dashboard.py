"""
Export all signal + market data to JSON for the signals dashboard.
"""

import json
import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/quant_signals.duckdb")
EXPORT_DIR = Path("data/export")


def _to_json(df, path):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].round(4)
    records = df.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and (v != v):
                rec[k] = None
    with open(path, "w") as f:
        json.dump(records, f)
    return len(records)


def export_all():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # ── Market Overview: latest 20 days equity top movers ──
    n = _to_json(con.execute("""
        WITH latest AS (
            SELECT MAX(trade_date) as max_date FROM equity_daily
        ),
        recent AS (
            SELECT e.trade_date, e.symbol, e.close, e.volume, e.delivery_pct, e.delivery_qty,
                   e.open, e.high, e.low
            FROM equity_daily e, latest l
            WHERE e.trade_date >= l.max_date - INTERVAL 30 DAY
              AND e.series = 'EQ'
              AND e.volume > 100000
        )
        SELECT * FROM recent ORDER BY trade_date DESC, volume DESC
    """).fetchdf(), EXPORT_DIR / "equity_recent.json")
    print(f"  equity_recent: {n} rows")

    # ── Signal 1: CADP — latest earnings events with scores ──
    cadp = pd.read_parquet("data/cadp_signal.parquet")
    cadp = cadp.sort_values("earnings_date", ascending=False)
    n = _to_json(cadp, EXPORT_DIR / "cadp_signal.json")
    print(f"  cadp_signal: {n} rows")

    # ── Signal 2: F&O — latest 5 days per symbol ──
    fo = pd.read_parquet("data/fo_signals.parquet")
    latest_fo_date = fo["trade_date"].max()
    fo_recent = fo[fo["trade_date"] >= latest_fo_date - pd.Timedelta(days=7)]
    n = _to_json(fo_recent, EXPORT_DIR / "fo_signals_recent.json")
    print(f"  fo_signals_recent: {n} rows")

    # F&O latest snapshot (one row per symbol, latest day)
    fo_latest = fo[fo["trade_date"] == latest_fo_date].sort_values("fo_bullish_raw", ascending=False)
    n = _to_json(fo_latest, EXPORT_DIR / "fo_latest.json")
    print(f"  fo_latest: {n} rows")

    # Participant divergence
    div = pd.read_parquet("data/participant_divergence.parquet")
    n = _to_json(div, EXPORT_DIR / "participant_divergence.json")
    print(f"  participant_divergence: {n} rows")

    # ── Signal 3: CHI — latest quarter per symbol ──
    chi = pd.read_parquet("data/chi_signal.parquet")
    chi_latest = chi.sort_values("quarter_end").groupby("symbol").tail(1).sort_values("chi", ascending=False)
    n = _to_json(chi_latest, EXPORT_DIR / "chi_latest.json")
    print(f"  chi_latest: {n} rows")

    # CHI full history for time series
    n = _to_json(chi, EXPORT_DIR / "chi_full.json")
    print(f"  chi_full: {n} rows")

    # ── Signal 4: Macro ──
    macro = pd.read_parquet("data/macro_nowcast.parquet")
    n = _to_json(macro, EXPORT_DIR / "macro_monthly.json")
    print(f"  macro_monthly: {n} rows")

    # Sector momentum from IIP
    from src.signals.macro_nowcast import compute_sector_momentum
    sectors = compute_sector_momentum(con)
    n = _to_json(sectors, EXPORT_DIR / "sector_momentum.json")
    print(f"  sector_momentum: {n} rows")

    # ── Corporate data ──
    # Insider trading: biggest recent transactions
    n = _to_json(con.execute("""
        SELECT security_name, person_name, category, transaction_type,
               acquired_qty, acquired_value, date_from, mode_of_acquisition
        FROM insider_trading
        WHERE date_from >= CURRENT_DATE - INTERVAL 90 DAY
          AND acquired_value > 0
        ORDER BY acquired_value DESC
        LIMIT 500
    """).fetchdf(), EXPORT_DIR / "insider_recent.json")
    print(f"  insider_recent: {n} rows")

    # Upcoming board meetings (next 30 days from latest data)
    n = _to_json(con.execute("""
        WITH latest AS (SELECT MAX(meeting_date) as max_date FROM board_meetings)
        SELECT meeting_date, symbol, company_name, purpose
        FROM board_meetings, latest
        WHERE meeting_date >= max_date - INTERVAL 30 DAY
          AND purpose LIKE '%Financial Result%'
        ORDER BY meeting_date DESC
    """).fetchdf(), EXPORT_DIR / "upcoming_earnings.json")
    print(f"  upcoming_earnings: {n} rows")

    # ── Signal 5: SUE ──
    sue_path = Path("data/sue_signal.parquet")
    if sue_path.exists():
        sue = pd.read_parquet(sue_path)
        n = _to_json(sue, EXPORT_DIR / "sue_signal.json")
        print(f"  sue_signal: {n} rows")

    # ── Signal 7: Momentum ──
    mom_path = Path("data/momentum_signal.parquet")
    if mom_path.exists():
        mom = pd.read_parquet(mom_path)
        mom = mom[mom["mom_score"].notna()].sort_values("mom_score", ascending=False)
        n = _to_json(mom, EXPORT_DIR / "momentum_signal.json")
        print(f"  momentum_signal: {n} rows")

    # ── Signal 8: Value + Quality ──
    vq_path = Path("data/value_quality_signal.parquet")
    if vq_path.exists():
        vq = pd.read_parquet(vq_path)
        vq = vq[vq["vq_score"].notna()].sort_values("vq_score", ascending=False)
        n = _to_json(vq, EXPORT_DIR / "value_quality_signal.json")
        print(f"  value_quality_signal: {n} rows")

    # ── Signal 9: Smart-money (bulk/block deals) ──
    sm_path = Path("data/smart_money_signal.parquet")
    if sm_path.exists():
        sm = pd.read_parquet(sm_path)
        n = _to_json(sm, EXPORT_DIR / "smart_money_signal.json")
        print(f"  smart_money_signal: {n} rows")
    smd_path = Path("data/smart_money_deals.parquet")
    if smd_path.exists():
        smd = pd.read_parquet(smd_path).head(150)
        n = _to_json(smd, EXPORT_DIR / "smart_money_deals.json")
        print(f"  smart_money_deals: {n} rows")

    # ── Signal 6: MasterScore ──
    ms_path = Path("data/master_score.parquet")
    if ms_path.exists():
        ms = pd.read_parquet(ms_path)
        n = _to_json(ms, EXPORT_DIR / "master_score.json")
        print(f"  master_score: {n} rows")

    # ── Conviction Score (unified composite) ──
    cs_path = Path("data/conviction_score.parquet")
    if cs_path.exists():
        cs = pd.read_parquet(cs_path)
        cs = cs[cs["final_score"].notna()].sort_values("final_score", ascending=False)
        n = _to_json(cs, EXPORT_DIR / "conviction_score.json")
        print(f"  conviction_score: {n} rows")

    # ── Summary stats ──
    stats = con.execute("""
        SELECT
            (SELECT COUNT(*) FROM equity_daily) as equity_rows,
            (SELECT COUNT(*) FROM fo_daily) as fo_rows,
            (SELECT COUNT(*) FROM insider_trading) as insider_rows,
            (SELECT MAX(trade_date) FROM equity_daily) as latest_equity_date,
            (SELECT MAX(trade_date) FROM fo_daily) as latest_fo_date,
            (SELECT COUNT(DISTINCT symbol) FROM equity_daily) as equity_symbols,
            (SELECT COUNT(DISTINCT symbol) FROM fo_daily WHERE instrument='FUTSTK') as fo_symbols
    """).fetchdf()
    _to_json(stats, EXPORT_DIR / "db_stats.json")
    print(f"  db_stats: 1 row")

    con.close()
    print(f"\nAll exports in {EXPORT_DIR.resolve()}")


if __name__ == "__main__":
    export_all()
