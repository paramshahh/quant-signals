"""
Bitemporal Point-in-Time (PIT) bridge — blueprint #2.

The backtester must never see a fundamental number before it was actually public.
This builds `fundamental_bridge`: every quarterly_results row tagged with a knowledge_date
(when the market learned it), so an ASOF JOIN can serve the correct vintage per trade_date.

Each fundamental row carries TWO timestamps (bitemporal):
  - effective_date : the period it covers (quarter end, e.g. 2026-03-31)
  - knowledge_date : when it became public (the board meeting that adopted the results)

THREE guardrails against engineering a new look-ahead bias:

  1. T+1 EXECUTION — board meetings often conclude after the 3:30pm close, so a result
     adopted on day T can only trade from T+1. The serving join is therefore STRICT:
         ASOF JOIN ... ON p.trade_date > b.knowledge_date          (NOT >=)

  2. 45/60-DAY FALLBACK — if the board-meeting date is missing (scrape gaps), we do NOT
     drop the quarter (that would orphan the stock from coverage shrinkage). We COALESCE to
     the SEBI statutory deadline: quarter_end + 45 days (regular quarter) or + 60 days for
     the March/annual quarter. Conservative — the data enters late, never early.

  3. INSERT-ONLY (restatement safe) — we never UPDATE history. If a later quarter restates an
     earlier one, a NEW row is appended with the same effective_date but a new knowledge_date
     (the later board meeting). ASOF JOIN then auto-serves the version known as of each
     trade_date. The table is keyed (symbol, effective_date, knowledge_date); re-runs only
     append rows not already present.

Serving query (the contract for the validation engine #4):
    SELECT p.trade_date, p.symbol, b.eps, b.net_profit
    FROM daily p
    ASOF JOIN fundamental_bridge b
      ON p.symbol = b.symbol AND p.trade_date > b.knowledge_date
"""

import argparse
import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "quant_signals.duckdb"
OUT_PARQUET = Path(__file__).resolve().parents[2] / "data" / "fundamental_bridge.parquet"

# window after quarter-end in which to look for the declaring board meeting
_MATCH_WINDOW_DAYS = 100

_DDL = """
CREATE TABLE IF NOT EXISTS fundamental_bridge (
    symbol            VARCHAR,
    effective_date    DATE,
    knowledge_date    DATE,
    knowledge_source  VARCHAR,
    eps               DOUBLE,
    sales             DOUBLE,
    operating_profit  DOUBLE,
    opm_pct           DOUBLE,
    net_profit        DOUBLE,
    loaded_at         TIMESTAMP
)
"""


def compute_bridge(con) -> pd.DataFrame:
    """Map each quarterly_results row to a knowledge_date (board meeting or SEBI fallback)."""
    df = con.execute(f"""
        WITH q AS (
            SELECT symbol,
                   last_day(CAST(quarter AS DATE))           AS effective_date,
                   eps, sales, operating_profit, opm_pct, net_profit
            FROM quarterly_results
            WHERE quarter IS NOT NULL
        ),
        bm AS (
            SELECT symbol, meeting_date
            FROM board_meetings
            WHERE meeting_date IS NOT NULL AND LOWER(purpose) LIKE '%result%'
        ),
        -- earliest results board meeting strictly after each quarter-end, within the window
        matched AS (
            SELECT q.symbol, q.effective_date, MIN(b.meeting_date) AS bm_date
            FROM q
            LEFT JOIN bm b
              ON b.symbol = q.symbol
             AND b.meeting_date > q.effective_date
             AND b.meeting_date <= q.effective_date + INTERVAL ({_MATCH_WINDOW_DAYS}) DAY
            GROUP BY q.symbol, q.effective_date
        )
        SELECT
            q.symbol,
            q.effective_date,
            -- Guardrail 2: COALESCE to SEBI deadline (60d for March/annual, else 45d)
            COALESCE(
                m.bm_date,
                q.effective_date + INTERVAL (CASE WHEN month(q.effective_date) = 3 THEN 60 ELSE 45 END) DAY
            ) AS knowledge_date,
            CASE WHEN m.bm_date IS NOT NULL THEN 'board_meeting' ELSE 'fallback_sebi' END AS knowledge_source,
            q.eps, q.sales, q.operating_profit, q.opm_pct, q.net_profit,
            now() AS loaded_at
        FROM q
        JOIN matched m ON m.symbol = q.symbol AND m.effective_date = q.effective_date
    """).fetchdf()
    return df


def build_pit_bridge(con=None) -> int:
    should_close = con is None
    if con is None:
        con = duckdb.connect(str(DB_PATH))

    con.execute(_DDL)
    bridge = compute_bridge(con)

    # Guardrail 3: INSERT-ONLY. Append only (symbol, effective_date, knowledge_date) combos
    # not already present — re-runs are idempotent; a restatement (new knowledge_date for the
    # same effective_date) appends a new vintage rather than overwriting history.
    con.register("bridge_df", bridge)
    before = con.execute("SELECT COUNT(*) FROM fundamental_bridge").fetchone()[0]
    con.execute("""
        INSERT INTO fundamental_bridge
        SELECT d.* FROM bridge_df d
        WHERE NOT EXISTS (
            SELECT 1 FROM fundamental_bridge fb
            WHERE fb.symbol = d.symbol
              AND fb.effective_date = d.effective_date
              AND fb.knowledge_date = d.knowledge_date
        )
    """)
    after = con.execute("SELECT COUNT(*) FROM fundamental_bridge").fetchone()[0]
    con.unregister("bridge_df")

    inserted = after - before
    bm = int((bridge["knowledge_source"] == "board_meeting").sum())
    fb = int((bridge["knowledge_source"] == "fallback_sebi").sum())
    print(f"  fundamental_bridge: {after} total rows (+{inserted} this run); "
          f"of the current mapping {bm} via board meeting, {fb} via SEBI fallback "
          f"({100*bm/max(bm+fb,1):.0f}% real dates)")

    con.execute(f"COPY (SELECT * FROM fundamental_bridge) TO '{OUT_PARQUET}' (FORMAT PARQUET)")
    if should_close:
        con.close()
    return inserted


def validate(symbol: str):
    """Demonstrate the no-look-ahead contract: which quarter is visible on which trade_date."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    print(f"\n=== {symbol}: bitemporal rows (effective vs knowledge) ===")
    print(con.execute("""
        SELECT effective_date, knowledge_date, knowledge_source, eps, net_profit
        FROM fundamental_bridge WHERE symbol = ?
        ORDER BY effective_date DESC LIMIT 6
    """, [symbol]).fetchdf().to_string(index=False))

    # ASOF: what EPS would the backtester legitimately know on three straddling dates?
    print(f"\n=== PIT ASOF JOIN (T+1 strict): EPS visible to the backtester ===")
    kd = con.execute("SELECT MAX(knowledge_date) FROM fundamental_bridge WHERE symbol=?", [symbol]).fetchone()[0]
    d_before = (pd.Timestamp(kd) - pd.Timedelta(days=2)).date().isoformat()
    d_on = pd.Timestamp(kd).date().isoformat()
    d_after = (pd.Timestamp(kd) + pd.Timedelta(days=2)).date().isoformat()
    q = f"""
        WITH probe(trade_date) AS (
            VALUES (DATE '{d_before}'), (DATE '{d_on}'), (DATE '{d_after}')
        )
        SELECT p.trade_date,
               b.effective_date AS latest_known_quarter,
               b.knowledge_date,
               b.eps
        FROM probe p
        ASOF LEFT JOIN fundamental_bridge b
          ON b.symbol = '{symbol}' AND p.trade_date > b.knowledge_date
        ORDER BY p.trade_date
    """
    print(f"  (most recent knowledge_date for {symbol} = {kd}; meeting day itself must NOT be visible)")
    print(con.execute(q).fetchdf().to_string(index=False))
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=str, help="Demonstrate the PIT contract for a symbol")
    args = ap.parse_args()
    build_pit_bridge()
    if args.validate:
        validate(args.validate.upper())
