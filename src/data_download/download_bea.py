"""
Download BEA data for Baqaee-Farhi replication.

Data needed:
  1. BEA Annual Industry Accounts — Use Table (71 sectors), years 2017–2022
  2. BEA GDP by Industry (Gross Output), years 2017–2022
  3. BEA Personal Consumption Expenditures by type, years 2019–2022

API documentation: https://apps.bea.gov/api/
Register for a free key at: https://apps.bea.gov/api/signup/

Usage:
    export BEA_API_KEY="your-key-here"
    python src/data_download/download_bea.py
"""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path

RAW_BEA_DIR = Path("data/raw/bea")
RAW_BEA_DIR.mkdir(parents=True, exist_ok=True)

BEA_API_BASE = "https://apps.bea.gov/api/data"


def get_bea_api_key() -> str:
    key = os.environ.get("BEA_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "BEA_API_KEY not set. Register at https://apps.bea.gov/api/signup/ "
            "and set: export BEA_API_KEY='your-key'"
        )
    return key


def bea_request(params: dict, retries: int = 3) -> dict:
    """Make a BEA API request with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.get(BEA_API_BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if "BEAAPI" in data and "Results" in data["BEAAPI"]:
                return data["BEAAPI"]["Results"]
            raise ValueError(f"Unexpected BEA response structure: {list(data.keys())}")
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  Attempt {attempt+1} failed ({e}), retrying in {wait}s...")
            time.sleep(wait)
    return {}


def download_use_table(api_key: str, year: int) -> pd.DataFrame:
    """
    Download BEA Input-Output Use Table for a given year.

    The 'InputOutput' dataset provides Use and Make tables.
    TableID 259 = Use of Commodities by Industries (Before Redefinitions), current $
    TableID 260 = Use of Commodities by Industries (After Redefinitions), current $
    """
    print(f"Downloading BEA Use Table for {year}...")
    params = {
        "UserID": api_key,
        "method": "GetData",
        "DataSetName": "InputOutput",
        "TableID": "259",  # Use table, before redefinitions
        "Year": str(year),
        "ResultFormat": "JSON",
    }
    results = bea_request(params)

    # Parse the response into a DataFrame
    if "Data" not in results:
        print(f"  Warning: No data returned for {year}")
        return pd.DataFrame()

    rows = results["Data"]
    df = pd.DataFrame(rows)
    out_path = RAW_BEA_DIR / f"use_table_{year}.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path}")
    return df


def download_gdp_by_industry(api_key: str, years: list) -> pd.DataFrame:
    """
    Download BEA GDP by Industry — Gross Output series.

    Dataset: GDPbyIndustry
    TableID: 1 = Value Added; 6 = Gross Output
    Frequency: A (annual)
    """
    print("Downloading BEA GDP by Industry (Gross Output)...")
    year_str = ",".join(str(y) for y in years)
    params = {
        "UserID": api_key,
        "method": "GetData",
        "DataSetName": "GDPbyIndustry",
        "TableID": "6",
        "Frequency": "A",
        "Year": year_str,
        "Industry": "ALL",
        "ResultFormat": "JSON",
    }
    results = bea_request(params)

    if "Data" not in results:
        print("  Warning: No GDP by Industry data returned")
        return pd.DataFrame()

    df = pd.DataFrame(results["Data"])
    out_path = RAW_BEA_DIR / "gdp_by_industry_gross_output.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path}")
    return df


def download_gdp_by_industry_value_added(api_key: str, years: list) -> pd.DataFrame:
    """Download value added by industry (TableID 1)."""
    print("Downloading BEA GDP by Industry (Value Added)...")
    year_str = ",".join(str(y) for y in years)
    params = {
        "UserID": api_key,
        "method": "GetData",
        "DataSetName": "GDPbyIndustry",
        "TableID": "1",
        "Frequency": "A",
        "Year": year_str,
        "Industry": "ALL",
        "ResultFormat": "JSON",
    }
    results = bea_request(params)

    if "Data" not in results:
        print("  Warning: No Value Added data returned")
        return pd.DataFrame()

    df = pd.DataFrame(results["Data"])
    out_path = RAW_BEA_DIR / "gdp_by_industry_value_added.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path}")
    return df


def download_nipa_gdp(api_key: str, years: list) -> pd.DataFrame:
    """
    Download NIPA Table 1.1.5 — GDP at quarterly frequency.
    Used to verify the Feb–May 2020 GDP decline.
    """
    print("Downloading NIPA GDP (quarterly)...")
    year_str = ",".join(str(y) for y in years)
    params = {
        "UserID": api_key,
        "method": "GetData",
        "DataSetName": "NIPA",
        "TableName": "T10105",  # Table 1.1.5 Gross Domestic Product
        "Frequency": "Q",
        "Year": year_str,
        "ResultFormat": "JSON",
    }
    results = bea_request(params)

    if "Data" not in results:
        print("  Warning: No NIPA GDP data returned")
        return pd.DataFrame()

    df = pd.DataFrame(results["Data"])
    out_path = RAW_BEA_DIR / "nipa_gdp_quarterly.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path}")
    return df


def download_pce_by_type(api_key: str, years: list) -> pd.DataFrame:
    """
    Download NIPA Table 2.4.5 — PCE by Type of Expenditure.
    Used for final demand shares (α_i^f).
    """
    print("Downloading PCE by Type of Expenditure...")
    year_str = ",".join(str(y) for y in years)
    params = {
        "UserID": api_key,
        "method": "GetData",
        "DataSetName": "NIPA",
        "TableName": "T20405",  # Table 2.4.5 PCE by Type
        "Frequency": "A",
        "Year": year_str,
        "ResultFormat": "JSON",
    }
    results = bea_request(params)

    if "Data" not in results:
        print("  Warning: No PCE data returned")
        return pd.DataFrame()

    df = pd.DataFrame(results["Data"])
    out_path = RAW_BEA_DIR / "pce_by_type.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path}")
    return df


def list_bea_datasets(api_key: str) -> None:
    """Helper: list all available BEA datasets."""
    params = {
        "UserID": api_key,
        "method": "GetDataSetList",
        "ResultFormat": "JSON",
    }
    results = bea_request(params)
    for ds in results.get("Dataset", []):
        print(f"  {ds['DatasetName']}: {ds['DatasetDescription']}")


def list_io_tables(api_key: str) -> None:
    """Helper: list available IO table IDs."""
    params = {
        "UserID": api_key,
        "method": "GetParameterValues",
        "DataSetName": "InputOutput",
        "ParameterName": "TableID",
        "ResultFormat": "JSON",
    }
    results = bea_request(params)
    for item in results.get("ParamValue", []):
        print(f"  TableID {item['Key']}: {item['Desc']}")


def main():
    api_key = get_bea_api_key()

    # IO tables: use 2017 as baseline structural year
    # Also download 2019–2022 for updating gross output
    io_years = [2017]
    output_years = list(range(2017, 2023))

    print("=" * 60)
    print("BEA Data Download")
    print("=" * 60)

    # 1. Download IO Use Tables
    for yr in io_years:
        df = download_use_table(api_key, yr)
        time.sleep(0.5)  # Be polite to the API

    # 2. Download Gross Output by Industry
    download_gdp_by_industry(api_key, output_years)
    time.sleep(0.5)

    # 3. Download Value Added by Industry
    download_gdp_by_industry_value_added(api_key, output_years)
    time.sleep(0.5)

    # 4. Download quarterly GDP (NIPA) for 2019–2022
    download_nipa_gdp(api_key, [2019, 2020, 2021, 2022])
    time.sleep(0.5)

    # 5. Download PCE by type
    download_pce_by_type(api_key, output_years)

    print("\nDownload complete. Files saved to data/raw/bea/")
    print("Next step: run src/io_network/construct_network.py")


if __name__ == "__main__":
    main()
