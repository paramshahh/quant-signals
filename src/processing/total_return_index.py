"""
Total Return Index / Corporate-Action adjustment engine.

We ingest RAW unadjusted NSE bhavcopy. A split/bonus/demerger puts an artificial gap in
the raw series that a naive momentum/volatility model reads as a crash (a 1:1 bonus looks
like -50%). This module builds a backward Cumulative Adjustment Factor (CAF) so every
downstream return calculation is clean.

DIRECTIONALITY (Trap 1): backward only. Today's raw price is the immutable anchor; each
ex-date factor is applied cumulatively to all STRICTLY EARLIER days:
    P_adj,t = P_raw,t * prod_{ex_date_i > t} factor_i

FACTORS:
  - Split  "From Rs X To Rs Y"  -> Y / X           (face-value ratio; handles reverse-split)
  - Bonus  A:B                  -> B / (A + B)      (EXCLUDES NCRPS/preference bonuses)
  - Dividend  Rs D per share    -> 1 - D / cum_close   (Trap 2: CUM-date close = day BEFORE ex,
                                                        NOT ex-date open; matches NSE TR method)
  - Demerger / scheme of arr.   -> implied = ex_open / prev_close   (Trap 5: pre-open discovered
                                    price proxy; override-able with published CoA ratio)

COLLISION HANDLING (Trap 3): all actions on the same ex-date are aggregated into ONE unified
daily factor per security BEFORE the cumulative product (e.g. same-day bonus + split = 0.25).

SYMBOL DRIFT (Trap 4): the time series is grouped by ISIN, not the ticker string. ISIN is the
permanent security identity; tickers are display labels and NSE renames them (CADILAHC->ZYDUSLIFE,
DVL<->DTIL). Grouping by ISIN bridges the rename so pre-rename history is still adjusted.

Output: data/adjusted_prices.parquet + DuckDB table `adjusted_prices`
        (per-row: trade_date, symbol as-traded, isin, canonical_symbol, close, adj_close,
         tri_close, caf_price, caf_tri, open/high/low adj, turnover, demerger_adj).
"""

import re
import argparse
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "quant_signals.duckdb"
OUT_PARQUET = Path(__file__).resolve().parents[2] / "data" / "adjusted_prices.parquet"

_NON_EQUITY = re.compile(r"ncrps|ncd|preference|pref\b|warrant|debenture", re.I)
_SPLIT_RE = re.compile(r"from\s+(?:rs|re)\.?\s*([\d.]+).*?to\s+(?:rs|re)\.?\s*([\d.]+)", re.I | re.S)
_BONUS_RE = re.compile(r"bonus\b[^0-9]*(\d+)\s*:\s*(\d+)", re.I)
_DIV_RE = re.compile(r"(?:rs|re)\.?\s*([\d.]+)\s*per\s*share", re.I)

# implausibility guards for the implied demerger factor (pure spin-offs drop the parent)
_DEMERGER_MIN, _DEMERGER_MAX = 0.30, 1.02


def parse_action(subject: str):
    """Return (kind, value) or None.
    kind 'mult'     -> value is a direct price multiplier (split/bonus)
    kind 'div'      -> value is per-share cash amount (needs cum-close)
    kind 'demerger' -> value is None (implied factor computed from prices)
    """
    if not subject:
        return None
    s = subject.strip()
    low = s.lower()

    if "demerger" in low or "scheme of arrangement" in low or "composite scheme" in low:
        return ("demerger", None)

    if "split" in low or "sub-division" in low or "sub division" in low or "consolidation" in low:
        m = _SPLIT_RE.search(s)
        if m:
            old_fv, new_fv = float(m.group(1)), float(m.group(2))
            if old_fv > 0:
                return ("mult", new_fv / old_fv)

    if "bonus" in low and not _NON_EQUITY.search(low):
        m = _BONUS_RE.search(s)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            if (a + b) > 0:
                return ("mult", b / (a + b))

    if "dividend" in low:
        amts = [float(x) for x in _DIV_RE.findall(s)]
        if amts:
            return ("div", float(sum(amts)))

    return None


def _build_isin_map(con) -> dict:
    """symbol -> ISIN, from corporate filings (the only tables carrying ISIN).
    Picks the modal ISIN per symbol. Maps BOTH sides of a rename to the same ISIN."""
    df = con.execute("""
        SELECT symbol, isin FROM corporate_announcements WHERE isin IS NOT NULL AND isin <> ''
        UNION ALL
        SELECT symbol, isin FROM corporate_actions WHERE isin IS NOT NULL AND isin <> ''
    """).fetchdf()
    if df.empty:
        return {}
    mode = (df.groupby(["symbol", "isin"]).size().reset_index(name="n")
              .sort_values("n", ascending=False).drop_duplicates("symbol"))
    return dict(zip(mode["symbol"], mode["isin"]))


def build_adjusted_prices(con=None) -> pd.DataFrame:
    should_close = con is None
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)

    prices = con.execute("""
        SELECT trade_date, symbol, open, high, low, close, turnover
        FROM equity_daily
        WHERE series = 'EQ' AND close > 0
        ORDER BY symbol, trade_date
    """).fetchdf()
    actions = con.execute("""
        SELECT symbol, ex_date, subject, isin
        FROM corporate_actions
        WHERE ex_date IS NOT NULL AND subject IS NOT NULL
    """).fetchdf()
    isin_map = _build_isin_map(con)
    if should_close:
        con.close()

    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    actions["ex_date"] = pd.to_datetime(actions["ex_date"])

    # Trap 4: assign a permanent key. Real ISIN where known, else synthetic per-ticker.
    def key_of(sym):
        return isin_map.get(sym, f"NOISIN:{sym}")
    prices["isin"] = prices["symbol"].map(key_of)

    # Parse actions and attach them to the SAME ISIN key as the prices.
    parsed = []
    for r in actions.itertuples(index=False):
        p = parse_action(r.subject)
        if p:
            parsed.append((key_of(r.symbol), r.ex_date, p[0], p[1]))
    pa = pd.DataFrame(parsed, columns=["isin", "ex_date", "kind", "value"])
    print(f"  Parsed {len(pa)} actionable corporate actions "
          f"({(pa.kind=='mult').sum()} split/bonus, {(pa.kind=='div').sum()} dividend, "
          f"{(pa.kind=='demerger').sum()} demerger)")
    actions_by_isin = {k: g for k, g in pa.groupby("isin")}

    n_renames = (prices.groupby("isin")["symbol"].nunique() > 1).sum()
    print(f"  {prices['isin'].nunique()} securities by ISIN; {n_renames} span a ticker rename (bridged)")

    out_frames = []
    n_adjusted = n_demerger = 0
    for isin, g in prices.groupby("isin", sort=False):
        g = g.sort_values("trade_date").drop_duplicates("trade_date").reset_index(drop=True)
        dates = g["trade_date"].to_numpy()
        close = g["close"].to_numpy(dtype=float)
        openp = g["open"].to_numpy(dtype=float)
        n = len(dates)
        canonical = g.iloc[-1]["symbol"]  # most recent ticker for this ISIN

        step_price = np.ones(n)
        step_tri = np.ones(n)
        demerger_hit = False

        acts = actions_by_isin.get(isin)
        if acts is not None:
            for a in acts.itertuples(index=False):
                ex = np.datetime64(a.ex_date)
                if ex <= dates[0] or ex > dates[-1]:
                    continue
                idx = int(np.searchsorted(dates, ex))
                if idx >= n:
                    continue
                if a.kind == "mult":
                    f = a.value
                    step_price[idx] *= f
                    step_tri[idx] *= f
                elif a.kind == "div":
                    if idx == 0:
                        continue
                    cum_close = close[idx - 1]          # Trap 2: prior-day close
                    if cum_close <= 0:
                        continue
                    f = 1.0 - (a.value / cum_close)
                    if 0 < f <= 1.0:
                        step_tri[idx] *= f               # dividends -> total-return series only
                elif a.kind == "demerger":
                    if idx == 0:
                        continue
                    prev_close = close[idx - 1]
                    ex_open = openp[idx]                 # Trap 5: pre-open discovered price proxy
                    if prev_close <= 0 or ex_open <= 0:
                        continue
                    f = ex_open / prev_close
                    if _DEMERGER_MIN <= f <= _DEMERGER_MAX:
                        step_price[idx] *= f
                        step_tri[idx] *= f
                        demerger_hit = True
                    # else: implausible -> leave unadjusted (likely needs published CoA ratio)

        def _caf(step):
            rev = np.cumprod(step[::-1])[::-1]   # prod_{j>=i}
            return rev / step                     # prod_{j>i}  (strictly after)

        caf_price = _caf(step_price)
        caf_tri = _caf(step_tri)
        if (caf_price != 1.0).any() or (caf_tri != 1.0).any():
            n_adjusted += 1
        if demerger_hit:
            n_demerger += 1

        out_frames.append(pd.DataFrame({
            "trade_date": dates,
            "symbol": g["symbol"].to_numpy(),
            "isin": isin,
            "canonical_symbol": canonical,
            "close": close,
            "adj_close": close * caf_price,
            "tri_close": close * caf_tri,
            "caf_price": caf_price,
            "caf_tri": caf_tri,
            "open": openp * caf_price,
            "high": g["high"].to_numpy() * caf_price,
            "low": g["low"].to_numpy() * caf_price,
            "turnover": g["turnover"].to_numpy(),
            "demerger_adj": demerger_hit,
        }))

    out = pd.concat(out_frames, ignore_index=True)
    print(f"  Adjusted {n_adjusted} securities ({n_demerger} had a demerger), {len(out)} rows")
    return out


def load_to_duckdb(df: pd.DataFrame):
    con = duckdb.connect(str(DB_PATH))
    con.execute("DROP TABLE IF EXISTS adjusted_prices")
    con.register("adj_df", df)
    con.execute("CREATE TABLE adjusted_prices AS SELECT * FROM adj_df")
    con.unregister("adj_df")
    con.close()
    print("  loaded DuckDB table: adjusted_prices")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=str, help="Show raw vs adjusted around a symbol's actions")
    args = ap.parse_args()

    df = build_adjusted_prices()
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"  saved {OUT_PARQUET.name}")
    load_to_duckdb(df)

    if args.validate:
        sym = args.validate.upper()
        sub = df[(df.symbol == sym) | (df.canonical_symbol == sym)].copy().sort_values("trade_date")
        sub["raw_ret"] = sub["close"].pct_change()
        sub["adj_ret"] = sub["adj_close"].pct_change()
        print(f"\n=== {sym}: ISIN(s) {sub['isin'].unique()}, tickers {sub['symbol'].unique()} ===")
        big = sub[sub["raw_ret"].abs() > 0.15]
        print("=== days with |raw daily return| > 15% (candidate phantom gaps) ===")
        print(big[["trade_date", "symbol", "close", "adj_close", "caf_price", "raw_ret", "adj_ret", "demerger_adj"]]
              .to_string(index=False))
