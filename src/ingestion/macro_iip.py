"""
IIP (Index of Industrial Production) Ingestion

eSankhyiki (esankhyiki.mospi.gov.in) is a JavaScript SPA with no API — cannot be automated.
This module provides a parser for manually downloaded IIP Excel files.

How to get the data:
  1. Go to https://esankhyiki.mospi.gov.in/macroindicators?product=iip
  2. Select indicators (General Index, Mining, Manufacturing, Electricity, etc.)
  3. Select date range
  4. Click "Download" (CSV or Excel)
  5. Save to data/raw/macro/iip/

The parser normalizes whatever file you drop in.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "macro" / "iip"


def parse_iip_file(path: Path) -> pd.DataFrame:
    """Parse an IIP Excel/CSV file into a normalized (period, sector, index_value) frame.

    Handles two layouts:
      A) MoSPI eSankhyiki LONG format — columns like
         base_year, year, month, type, category, sub_category, index, growth_rate.
      B) Generic WIDE format — first/period column is the date, remaining columns
         are sectors (fallback for ad-hoc downloads).
    """
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    cols = {str(c).lower(): c for c in df.columns}

    # ── Layout A: MoSPI long format ──
    if {"year", "month", "index"}.issubset(cols):
        y = df[cols["year"]].astype(str).str.strip()
        m = df[cols["month"]].astype(str).str.strip()
        # month may be a full name ("March") or abbreviation ("Mar"); coerce both
        period = pd.to_datetime(y + " " + m, format="%Y %B", errors="coerce")
        miss = period.isna()
        if miss.any():
            period[miss] = pd.to_datetime(y[miss] + " " + m[miss], format="%Y %b", errors="coerce")

        # sector granularity: most specific available (sub_category > category > type)
        sub = df[cols["sub_category"]] if "sub_category" in cols else None
        cat = df[cols["category"]] if "category" in cols else None
        typ = df[cols["type"]] if "type" in cols else None
        sector = sub if sub is not None else (cat if cat is not None else typ)
        if sub is not None and cat is not None:
            sector = sub.where(sub.notna(), cat)

        out = pd.DataFrame({
            "period": period,
            "sector": sector.astype(str).str.strip() if sector is not None else "General",
            "index_value": pd.to_numeric(df[cols["index"]], errors="coerce"),
        })
        out = out.dropna(subset=["period", "index_value"])
        return out

    # ── Layout B: generic wide fallback ──
    date_col = next((cols[k] for k in ("period", "month", "date", "time") if k in cols), df.columns[0])
    value_cols = [c for c in df.columns if c != date_col]
    melted = df.melt(id_vars=[date_col], value_vars=value_cols,
                     var_name="sector", value_name="index_value")
    melted = melted.rename(columns={date_col: "period"})
    melted["period"] = pd.to_datetime(melted["period"], errors="coerce")
    melted["index_value"] = pd.to_numeric(melted["index_value"], errors="coerce")
    melted = melted.dropna(subset=["period", "index_value"])
    return melted


def parse_all_iip_files() -> pd.DataFrame:
    """Parse all IIP files in the raw directory."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    files = list(RAW_DIR.glob("*.xlsx")) + list(RAW_DIR.glob("*.xls")) + list(RAW_DIR.glob("*.csv"))
    if not files:
        print(f"No IIP files found in {RAW_DIR}")
        print("Download from https://esankhyiki.mospi.gov.in/macroindicators?product=iip")
        return pd.DataFrame()

    all_dfs = []
    for f in files:
        try:
            df = parse_iip_file(f)
            if not df.empty:
                print(f"  {f.name}: {len(df)} records")
                all_dfs.append(df)
        except Exception as e:
            print(f"  {f.name}: parse error - {e}")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["period", "sector"])
    print(f"Total IIP records: {len(combined)}")
    return combined


if __name__ == "__main__":
    df = parse_all_iip_files()
    if not df.empty:
        print(df.head(20))
