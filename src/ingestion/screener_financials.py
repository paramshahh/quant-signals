"""
Scrape quarterly financials, balance sheet, key ratios, and shareholding
from screener.in for NSE-listed companies.

Fills critical data gaps for Signal 5 (SUE) and Signal 6 (MasterScore):
- Quarterly EPS → time-series earnings surprise
- Book value, shares outstanding → valuation signals
- Shareholding (FII/DII) → institutional flow cross-check

Usage:
    python -m src.ingestion.screener_financials          # scrape F&O + top 300
    python -m src.ingestion.screener_financials --test 5  # test with 5 symbols
"""

import argparse
import json
import re
import time
import random
import duckdb
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "quant_signals.duckdb"
EXPORT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "screener"

BASE_URL = "https://www.screener.in/company/{symbol}/consolidated/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

DELAY_RANGE = (0.4, 0.8)


def _parse_number(text):
    """Parse Indian number format: '2,12,834' → 212834, '15%' → 15."""
    if not text:
        return None
    text = text.strip().replace(",", "").replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def _parse_table(table):
    """Parse a screener.in HTML table into {row_label: [values]}."""
    if not table:
        return {}, []

    thead = table.find("thead")
    headers = [th.get_text(strip=True) for th in thead.find_all("th")] if thead else []

    rows = {}
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        label = tds[0].get_text(strip=True).rstrip("+")
        vals = [_parse_number(td.get_text(strip=True)) for td in tds[1:]]
        rows[label] = vals

    return rows, headers[1:]


def _parse_top_ratios(soup):
    """Parse the summary ratios from the top of the page."""
    top = soup.find(id="top")
    if not top:
        return {}

    ratios = {}
    for li in top.find_all("li"):
        spans = li.find_all("span")
        if len(spans) >= 2:
            key = spans[0].get_text(strip=True)
            val = spans[1].get_text(strip=True)
            ratios[key] = val
        else:
            text = li.get_text(strip=True)
            match = re.match(r"(.+?)([\d,.\-]+.*)", text)
            if match:
                ratios[match.group(1).strip()] = match.group(2).strip()

    return ratios


def scrape_company(symbol, session=None):
    """Scrape all financial data for one symbol. Returns dict or None on failure."""
    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)

    # Handle special chars: screener uses the NSE symbol directly in URL
    # but some need URL-encoding (& → %26) or alternate names
    SYMBOL_MAP = {
        "L&TFH": "L%26TFH",
        "M&M": "M%26M",
        "M&MFIN": "M%26MFIN",
        "J&KBANK": "J%26KBANK",
        "L&T": "L%26T",
    }
    url_sym = SYMBOL_MAP.get(symbol, symbol)

    url = BASE_URL.format(symbol=url_sym)
    try:
        r = session.get(url, timeout=20)
    except requests.RequestException as e:
        print(f"  {symbol}: request error: {e}")
        return None

    if r.status_code == 404:
        url_standalone = url.replace("/consolidated/", "/")
        try:
            r = session.get(url_standalone, timeout=20)
        except requests.RequestException:
            return None

    if r.status_code != 200:
        print(f"  {symbol}: HTTP {r.status_code}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    result = {"symbol": symbol}

    # Top ratios
    result["ratios"] = _parse_top_ratios(soup)

    # Quarterly results
    q_section = soup.find(id="quarters")
    if q_section:
        table = q_section.find("table")
        rows, periods = _parse_table(table)
        result["quarterly"] = {"periods": periods, **rows}

    # Profit & Loss (annual)
    pl_section = soup.find(id="profit-loss")
    if pl_section:
        table = pl_section.find("table")
        rows, periods = _parse_table(table)
        result["annual_pl"] = {"periods": periods, **rows}

    # Balance Sheet
    bs_section = soup.find(id="balance-sheet")
    if bs_section:
        table = bs_section.find("table")
        rows, periods = _parse_table(table)
        result["balance_sheet"] = {"periods": periods, **rows}

    # Shareholding (quarterly)
    sh_section = soup.find(id="shareholding")
    if sh_section:
        tables = sh_section.find_all("table")
        if tables:
            rows, periods = _parse_table(tables[0])
            result["shareholding_q"] = {"periods": periods, **rows}

    return result


def get_target_symbols(con, limit=None, top_n=1500):
    """Get F&O symbols + top equity by volume."""
    fo_symbols = con.execute(
        "SELECT DISTINCT symbol FROM fo_daily WHERE instrument = 'FUTSTK'"
    ).fetchdf()["symbol"].tolist()

    eq_top = con.execute(f"""
        SELECT symbol, SUM(volume) as total_vol
        FROM equity_daily
        WHERE series = 'EQ' AND trade_date >= CURRENT_DATE - INTERVAL 90 DAY
        GROUP BY symbol
        ORDER BY total_vol DESC
        LIMIT {top_n}
    """).fetchdf()["symbol"].tolist()

    combined = list(dict.fromkeys(fo_symbols + eq_top))
    if limit:
        combined = combined[:limit]
    return combined


def scrape_all(symbols, output_dir=EXPORT_DIR, force=False):
    """Scrape all symbols and save individual JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    success, fail, skip = 0, 0, 0

    for i, symbol in enumerate(symbols):
        out_file = output_dir / f"{symbol}.json"
        if out_file.exists() and not force:
            skip += 1
            continue

        print(f"[{i+1}/{len(symbols)}] {symbol}...", end=" ", flush=True)
        data = scrape_company(symbol, session)

        if data and "quarterly" in data:
            with open(out_file, "w") as f:
                json.dump(data, f)
            n_quarters = len(data["quarterly"].get("periods", []))
            print(f"OK ({n_quarters}Q)")
            success += 1
        else:
            print("FAIL")
            fail += 1

        delay = random.uniform(*DELAY_RANGE)
        time.sleep(delay)

    print(f"\nDone: {success} scraped, {skip} skipped (cached), {fail} failed")
    return success, skip, fail


def build_quarterly_df(data_dir=EXPORT_DIR):
    """Build a consolidated quarterly earnings DataFrame from scraped JSONs."""
    records = []
    for f in data_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        symbol = data["symbol"]
        q = data.get("quarterly", {})
        periods = q.get("periods", [])
        sales = q.get("Sales", [])
        net_profit = q.get("Net Profit", [])
        eps = q.get("EPS in Rs", [])
        opm = q.get("OPM %", [])
        op = q.get("Operating Profit", [])

        for i, period in enumerate(periods):
            records.append({
                "symbol": symbol,
                "quarter": period,
                "sales": sales[i] if i < len(sales) else None,
                "operating_profit": op[i] if i < len(op) else None,
                "opm_pct": opm[i] if i < len(opm) else None,
                "net_profit": net_profit[i] if i < len(net_profit) else None,
                "eps": eps[i] if i < len(eps) else None,
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df["quarter"] = pd.to_datetime(df["quarter"], format="%b %Y")
        df = df.sort_values(["symbol", "quarter"]).reset_index(drop=True)
    return df


def build_balance_sheet_df(data_dir=EXPORT_DIR):
    """Build annual balance sheet DataFrame."""
    records = []
    for f in data_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        symbol = data["symbol"]
        bs = data.get("balance_sheet", {})
        periods = bs.get("periods", [])

        for i, period in enumerate(periods):
            rec = {"symbol": symbol, "year": period}
            for key in ["Equity Capital", "Reserves", "Borrowings", "Total Liabilities",
                        "Fixed Assets", "CWIP", "Investments", "Other Assets", "Total Assets"]:
                vals = bs.get(key, [])
                rec[key.lower().replace(" ", "_")] = vals[i] if i < len(vals) else None
            records.append(rec)

    df = pd.DataFrame(records)
    if not df.empty:
        df["year"] = pd.to_datetime(df["year"], format="%b %Y")
        df = df.sort_values(["symbol", "year"]).reset_index(drop=True)
    return df


def build_ratios_df(data_dir=EXPORT_DIR):
    """Build key ratios DataFrame from top-of-page summary."""
    records = []
    for f in data_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        symbol = data["symbol"]
        ratios = data.get("ratios", {})
        records.append({
            "symbol": symbol,
            "market_cap_cr": _parse_number(ratios.get("Market Cap", "").replace("₹", "").replace("Cr.", "")),
            "current_price": _parse_number(ratios.get("Current Price", "").replace("₹", "")),
            "pe_ratio": _parse_number(ratios.get("Stock P/E", "")),
            "book_value": _parse_number(ratios.get("Book Value", "").replace("₹", "")),
            "dividend_yield": _parse_number(ratios.get("Dividend Yield", "").replace("%", "")),
            "roce": _parse_number(ratios.get("ROCE", "").replace("%", "")),
            "roe": _parse_number(ratios.get("ROE", "").replace("%", "")),
            "face_value": _parse_number(ratios.get("Face Value", "").replace("₹", "")),
        })

    return pd.DataFrame(records)


def load_to_duckdb(con=None):
    """Load all scraped data into DuckDB tables."""
    should_close = False
    if con is None:
        con = duckdb.connect(str(DB_PATH))
        should_close = True

    quarterly = build_quarterly_df()
    if not quarterly.empty:
        con.execute("DROP TABLE IF EXISTS quarterly_results")
        con.register("quarterly_df", quarterly)
        con.execute("CREATE TABLE quarterly_results AS SELECT * FROM quarterly_df")
        print(f"  quarterly_results: {len(quarterly)} rows, {quarterly['symbol'].nunique()} symbols")

    bs = build_balance_sheet_df()
    if not bs.empty:
        con.execute("DROP TABLE IF EXISTS balance_sheet")
        con.register("bs_df", bs)
        con.execute("CREATE TABLE balance_sheet AS SELECT * FROM bs_df")
        print(f"  balance_sheet: {len(bs)} rows")

    ratios = build_ratios_df()
    if not ratios.empty:
        con.execute("DROP TABLE IF EXISTS company_ratios")
        con.register("ratios_df", ratios)
        con.execute("CREATE TABLE company_ratios AS SELECT * FROM ratios_df")
        print(f"  company_ratios: {len(ratios)} rows")

    if should_close:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, help="Scrape only N symbols for testing")
    parser.add_argument("--load", action="store_true", help="Load cached JSONs into DuckDB (no scraping)")
    parser.add_argument("--force", action="store_true", help="Re-scrape even if cached (refresh quarterly data)")
    parser.add_argument("--top-n", type=int, default=1500, help="Top N equity symbols by volume (default: 1500)")
    args = parser.parse_args()

    if args.load:
        print("Loading cached data into DuckDB...")
        load_to_duckdb()
    else:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        symbols = get_target_symbols(con, limit=args.test, top_n=args.top_n)
        con.close()
        print(f"Targeting {len(symbols)} symbols (top {args.top_n} equity + F&O)")
        scrape_all(symbols, force=args.force)
        print("\nLoading into DuckDB...")
        load_to_duckdb()
