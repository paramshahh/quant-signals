"""
Signal 8: Value + Quality

Turns the screener.in fundamentals (company_ratios + balance_sheet +
quarterly_results) — previously only *displayed*, never *scored* — into two
cross-sectional factors and a combined composite.

VALUE  (cheap = good):
  - earnings_yield = 1 / P/E      (only when P/E > 0; negative earnings excluded)
  - book_to_price  = BookValue / Price   (inverse of P/B; high = cheap)
  - dividend_yield

QUALITY  (profitable, well-run, low-leverage, improving = good):
  - ROE, ROCE
  - debt_to_equity = Borrowings / (EquityCapital + Reserves)   (lower = better)
  - opm_trend      = avg OPM last 4 qtrs - avg OPM prior 4 qtrs (margin expansion)
  - profit_consistency = fraction of last 8 qtrs with positive net profit

Each component is z-scored cross-sectionally (winsorized at +/-3sigma), averaged
into value_score / quality_score, then combined 50/50 into vq_score. vq_score is
what feeds MasterScore as raw_vq.

The MUTHOOTFIN-style "cheap + accelerating" name we picked by hand falls straight
out of the top of this ranking automatically.

Output: data/value_quality_signal.parquet
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "quant_signals.duckdb"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _zwins(s: pd.Series, clip: float = 3.0) -> pd.Series:
    mu, sigma = s.mean(skipna=True), s.std(skipna=True)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sigma).clip(-clip, clip)


def _quality_from_fundamentals(con) -> pd.DataFrame:
    """Leverage (latest balance sheet) + margin trend & profit consistency
    (quarterly P&L)."""
    bs = con.execute("""
        SELECT symbol, year, equity_capital, reserves, borrowings
        FROM balance_sheet
    """).fetchdf()
    if not bs.empty:
        bs["year"] = pd.to_datetime(bs["year"])
        bs = bs.sort_values("year").groupby("symbol").tail(1)
        equity = (bs["equity_capital"].fillna(0) + bs["reserves"].fillna(0))
        bs["debt_to_equity"] = np.where(equity > 0, bs["borrowings"].fillna(0) / equity, np.nan)
        bs = bs[["symbol", "debt_to_equity"]]

    q = con.execute("""
        SELECT symbol, quarter, opm_pct, net_profit
        FROM quarterly_results
        ORDER BY symbol, quarter
    """).fetchdf()

    rows = []
    for symbol, g in q.groupby("symbol"):
        g = g.sort_values("quarter")
        opm = g["opm_pct"].to_numpy(dtype=float)
        npr = g["net_profit"].to_numpy(dtype=float)
        opm_trend = np.nan
        if len(opm) >= 8 and not np.any(np.isnan(opm[-8:])):
            opm_trend = opm[-4:].mean() - opm[-8:-4].mean()
        consistency = np.nan
        if len(npr) >= 8:
            last8 = npr[-8:]
            consistency = np.mean(last8 > 0)
        rows.append({"symbol": symbol, "opm_trend": opm_trend, "profit_consistency": consistency})
    qual = pd.DataFrame(rows)

    if bs is None or bs.empty:
        return qual
    return qual.merge(bs, on="symbol", how="outer")


def compute_value_quality(
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    if con is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        should_close = True
    else:
        should_close = False

    ratios = con.execute("""
        SELECT symbol, market_cap_cr, current_price, pe_ratio, book_value,
               dividend_yield, roce, roe
        FROM company_ratios
    """).fetchdf()

    qual = _quality_from_fundamentals(con)

    if should_close:
        con.close()

    if ratios.empty:
        return pd.DataFrame()

    df = ratios.merge(qual, on="symbol", how="left")

    # ── Value components ──
    df["earnings_yield"] = np.where(df["pe_ratio"] > 0, 1.0 / df["pe_ratio"], np.nan)
    df["book_to_price"] = np.where(
        df["current_price"] > 0, df["book_value"] / df["current_price"], np.nan
    )
    df["div_yield"] = df["dividend_yield"]

    value_parts = pd.DataFrame({
        "ey": _zwins(df["earnings_yield"]),
        "bp": _zwins(df["book_to_price"]),
        "dy": _zwins(df["div_yield"]),
    })
    df["value_score"] = value_parts.mean(axis=1)

    # ── Quality components ──
    quality_parts = pd.DataFrame({
        "roe": _zwins(df["roe"]),
        "roce": _zwins(df["roce"]),
        "lev": -_zwins(df["debt_to_equity"]),  # less debt = higher quality
        "opm": _zwins(df["opm_trend"]),
        "cons": _zwins(df["profit_consistency"]),
    })
    df["quality_score"] = quality_parts.mean(axis=1)

    # ── Combined ──
    df["vq_score"] = df[["value_score", "quality_score"]].mean(axis=1)

    df["value_rank"] = df["value_score"].rank(pct=True)
    df["quality_rank"] = df["quality_score"].rank(pct=True)
    df["vq_rank"] = df["vq_score"].rank(pct=True)

    df = df.sort_values("vq_score", ascending=False, na_position="last").reset_index(drop=True)
    print(f"  Value+Quality computed for {len(df)} symbols "
          f"(value n={df['value_score'].notna().sum()}, quality n={df['quality_score'].notna().sum()})")
    return df


if __name__ == "__main__":
    vq = compute_value_quality()
    if not vq.empty:
        cols = ["symbol", "pe_ratio", "roe", "roce", "debt_to_equity",
                "opm_trend", "value_score", "quality_score", "vq_score"]
        print(f"\n=== Value+Quality Top 20 (best combined) ===")
        print(vq.head(20)[cols].to_string(index=False))
        print(f"\n=== Cheapest 10 (value_score) ===")
        print(vq.sort_values("value_score", ascending=False).head(10)[
            ["symbol", "pe_ratio", "book_to_price", "div_yield", "value_score"]].to_string(index=False))
        print(f"\n=== Highest quality 10 ===")
        print(vq.sort_values("quality_score", ascending=False).head(10)[
            ["symbol", "roe", "roce", "debt_to_equity", "opm_trend", "quality_score"]].to_string(index=False))

        out = DATA_DIR / "value_quality_signal.parquet"
        vq.to_parquet(out, index=False)
        print(f"\nSaved {len(vq)} rows to {out.name}")
