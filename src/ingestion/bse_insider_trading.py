"""
BSE Insider Trading (PIT) Scraper

BSE's Akamai WAF blocks headless browsers entirely.
This script opens a VISIBLE Chrome window to scrape insider trading data.

Usage:
    python3 -m src.ingestion.bse_insider_trading --start 2023-01-01 --end 2025-05-16

The script will:
  1. Open Chrome (visible window)
  2. Navigate to BSE homepage — WAIT for you to clear any Akamai challenge
  3. Navigate to insider trading page
  4. Set date range and submit
  5. Click "Download All Records" to get CSV
  6. Parse and save to data/raw/corporate/insider_trading/

Run this at your desk — it needs a visible Chrome window and may need you to solve a CAPTCHA.
"""

import os
import re
import csv
import glob
import json
import time
import shutil
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "corporate" / "insider_trading"

CHROMEDRIVER = Path.home() / ".cache" / "selenium" / "chromedriver" / "mac-arm64" / "148.0.7778.167" / "chromedriver"
BSE_URL = "https://www.bseindia.com/corporates/Insider_Trading_new.aspx"


def _get_driver(download_dir: str):
    """Create a visible Chrome driver with anti-detection measures."""
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium import webdriver

    opts = Options()
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1400,900")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
    })

    service = Service(executable_path=str(CHROMEDRIVER))
    driver = webdriver.Chrome(service=service, options=opts)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
    })
    return driver


def _wait_for_download(download_dir: str, timeout: int = 60) -> Optional[str]:
    """Wait for a file to appear in the download directory."""
    start = time.time()
    while time.time() - start < timeout:
        files = glob.glob(os.path.join(download_dir, "*.csv"))
        if files:
            newest = max(files, key=os.path.getctime)
            if not newest.endswith(".crdownload"):
                time.sleep(1)
                return newest
        time.sleep(1)
    return None


def _wait_for_element(driver, by, selector, timeout=60):
    """Wait for an element to appear, polling every 2s. Returns element or None."""
    from selenium.webdriver.common.by import By
    start = time.time()
    while time.time() - start < timeout:
        try:
            el = driver.find_element(by, selector)
            if el.is_displayed():
                return el
        except Exception:
            pass
        time.sleep(2)
    return None


def _wait_for_real_page(driver, timeout=120):
    """Wait until BSE serves the real page (not Akamai challenge).

    Prints a message asking the user to solve any CAPTCHA if needed.
    """
    from selenium.webdriver.common.by import By

    print("\n  Waiting for BSE page to load (solve CAPTCHA if one appears)...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            body = driver.find_element(By.TAG_NAME, "body").text.lower()
            if "access denied" in body or "please wait" in body or "checking your browser" in body:
                time.sleep(3)
                continue
            # Check if we have BSE content (nav menu, footer, etc.)
            if "bse" in body or "bombay stock exchange" in body or "insider" in body:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def scrape_insider_trading(
    start: date,
    end: date,
    chunk_days: int = 30,
) -> Path:
    """Scrape BSE insider trading data in date-range chunks."""
    from selenium.webdriver.common.by import By

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dl_dir = str(RAW_DIR / "_downloads")
    os.makedirs(dl_dir, exist_ok=True)

    driver = _get_driver(dl_dir)
    all_records = []

    try:
        # Step 1: Visit BSE homepage and wait for real page / CAPTCHA clearance
        print("Opening BSE homepage...")
        driver.get("https://www.bseindia.com/")
        if not _wait_for_real_page(driver, timeout=120):
            print("ERROR: BSE homepage didn't load after 2 minutes. Akamai may be blocking.")
            print("Try again later or clear cookies and retry.")
            return RAW_DIR

        print("BSE homepage loaded. Waiting 5s before proceeding...")
        time.sleep(5)

        current = start
        while current <= end:
            chunk_end = min(current + timedelta(days=chunk_days - 1), end)
            from_str = current.strftime("%d/%m/%Y")
            to_str = chunk_end.strftime("%d/%m/%Y")

            print(f"\n  Fetching {from_str} to {to_str}...")

            # Clear download dir
            for f in glob.glob(os.path.join(dl_dir, "*.csv")):
                os.remove(f)

            driver.get(BSE_URL)

            # Wait for the insider trading page to actually render
            if not _wait_for_real_page(driver, timeout=90):
                print("  Page didn't load. Akamai challenge? Waiting 30s for manual solve...")
                time.sleep(30)
                if not _wait_for_real_page(driver, timeout=60):
                    print("  Still blocked. Skipping this chunk.")
                    current = chunk_end + timedelta(days=1)
                    continue

            time.sleep(3)

            try:
                # Try multiple selector patterns for the date inputs
                from_input = None
                to_input = None

                # Dump page source to find actual IDs if selectors fail
                for from_sel in [
                    "input[id*='txtFromDt']", "input[id*='FromDt']", "input[id*='fromDt']",
                    "input[name*='txtFromDt']", "input[name*='FromDt']",
                    "#ContentPlaceHolder1_txtFromDt", "#ctl00_ContentPlaceHolder1_txtFromDt",
                ]:
                    from_input = _wait_for_element(driver, By.CSS_SELECTOR, from_sel, timeout=10)
                    if from_input:
                        break

                for to_sel in [
                    "input[id*='txtToDate']", "input[id*='ToDate']", "input[id*='toDate']",
                    "input[name*='txtToDate']", "input[name*='ToDate']",
                    "#ContentPlaceHolder1_txtToDate", "#ctl00_ContentPlaceHolder1_txtToDate",
                ]:
                    to_input = _wait_for_element(driver, By.CSS_SELECTOR, to_sel, timeout=10)
                    if to_input:
                        break

                if not from_input or not to_input:
                    # Last resort: find ALL visible text inputs
                    inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='text']")
                    visible = [i for i in inputs if i.is_displayed()]
                    print(f"  Could not find date inputs. Found {len(visible)} visible text inputs.")
                    if len(visible) >= 2:
                        # Dump their IDs for debugging
                        for i, inp in enumerate(visible):
                            print(f"    input[{i}]: id={inp.get_attribute('id')}, name={inp.get_attribute('name')}, placeholder={inp.get_attribute('placeholder')}")
                    else:
                        # Dump page title and URL for debugging
                        print(f"  Page title: {driver.title}")
                        print(f"  Page URL: {driver.current_url}")
                        body_text = driver.find_element(By.TAG_NAME, "body").text[:500]
                        print(f"  Body preview: {body_text[:200]}")
                    current = chunk_end + timedelta(days=1)
                    continue

                # Clear and set date values
                driver.execute_script("arguments[0].value = ''", from_input)
                driver.execute_script("arguments[0].value = ''", to_input)
                driver.execute_script(f"arguments[0].value = '{from_str}'", from_input)
                driver.execute_script(f"arguments[0].value = '{to_str}'", to_input)
                time.sleep(1)

                # Find and click submit — try multiple patterns
                submit_btn = None
                for btn_sel in [
                    "input[id*='btnSubmit']", "button[id*='btnSubmit']",
                    "input[value='Submit']", "input[type='submit']",
                    "#ContentPlaceHolder1_btnSubmit", "#ctl00_ContentPlaceHolder1_btnSubmit",
                ]:
                    submit_btn = _wait_for_element(driver, By.CSS_SELECTOR, btn_sel, timeout=5)
                    if submit_btn:
                        break

                if not submit_btn:
                    print("  Could not find submit button. Skipping chunk.")
                    current = chunk_end + timedelta(days=1)
                    continue

                submit_btn.click()
                print("  Submitted. Waiting for results...")
                time.sleep(10)

                # Try to find and click "Download All Records"
                dl_link = None
                for dl_sel in [
                    "Download All", "Download", "Export",
                ]:
                    try:
                        dl_link = driver.find_element(By.PARTIAL_LINK_TEXT, dl_sel)
                        if dl_link.is_displayed():
                            break
                        dl_link = None
                    except Exception:
                        dl_link = None

                if not dl_link:
                    # Try input/button with download-like text
                    for dl_btn_sel in [
                        "a[id*='lnkDownload']", "input[id*='btnDownload']",
                        "#ContentPlaceHolder1_lnkDownload",
                    ]:
                        dl_link = _wait_for_element(driver, By.CSS_SELECTOR, dl_btn_sel, timeout=5)
                        if dl_link:
                            break

                if dl_link:
                    dl_link.click()
                    time.sleep(5)

                    csv_path = _wait_for_download(dl_dir)
                    if csv_path:
                        df = pd.read_csv(csv_path)
                        all_records.append(df)
                        print(f"    Got {len(df)} records")
                    else:
                        print(f"    No CSV downloaded (timeout)")
                else:
                    # Maybe no results for this date range
                    body = driver.find_element(By.TAG_NAME, "body").text
                    if "no record" in body.lower() or "no data" in body.lower():
                        print(f"    No records for this date range")
                    else:
                        print(f"    Could not find download link")

            except Exception as e:
                print(f"    Error: {e}")

            current = chunk_end + timedelta(days=1)
            time.sleep(3)

    finally:
        driver.quit()

    if all_records:
        combined = pd.concat(all_records, ignore_index=True)
        combined = combined.drop_duplicates()
        out_path = RAW_DIR / f"insider_trading_{start.isoformat()}_{end.isoformat()}.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nTotal: {len(combined)} unique records -> {out_path.name}")
        shutil.rmtree(dl_dir, ignore_errors=True)
        return out_path
    else:
        print("\nNo records downloaded.")

    return RAW_DIR


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape BSE Insider Trading Data")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=str(date.today()), help="End date YYYY-MM-DD")
    parser.add_argument("--chunk-days", type=int, default=30, help="Days per chunk")
    args = parser.parse_args()

    scrape_insider_trading(
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        args.chunk_days,
    )
