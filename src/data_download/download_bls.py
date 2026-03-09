"""
Download BLS data for Baqaee-Farhi replication.

Data needed:
  1. Producer Price Index (PPI) by industry — monthly, 2019–2022
  2. Consumer Price Index (CPI) by expenditure category — monthly, 2019–2022
  3. Output by industry (BLS MFP program) — annual, 2017–2022

BLS Public Data API v2 does not require a key for small requests (<500 series).
Registration (free) allows larger requests: https://data.bls.gov/registrationEngine/

Series ID naming conventions:
  PPI:  PCU + 6-digit NAICS code (e.g., PCU311111311111 = Dog food)
  CPI:  CUSR0000SA0 = CPI-U All items; CUSR0000SAF = Food; CUSR0000SAE = Energy
  MFP:  MPU + series code

Usage:
    python src/data_download/download_bls.py
"""

import json
import time
import requests
import pandas as pd
from pathlib import Path

RAW_BLS_DIR = Path("data/raw/bls")
RAW_BLS_DIR.mkdir(parents=True, exist_ok=True)

BLS_API_V2 = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


# ─────────────────────────────────────────────────────────────
# CPI series IDs (CUSR0000 = U.S. city average, seasonally adjusted)
# ─────────────────────────────────────────────────────────────
CPI_SERIES = {
    "CPI_All": "CUSR0000SA0",
    "CPI_Food": "CUSR0000SAF1",
    "CPI_Food_Home": "CUSR0000SAF11",
    "CPI_Food_Away": "CUSR0000SEFV",
    "CPI_Energy": "CUSR0000SA0E",
    "CPI_Energy_Gasoline": "CUSR0000SETB01",
    "CPI_Energy_Utilities": "CUSR0000SAH2",
    "CPI_Commodities_ex_Food_Energy": "CUSR0000SACL1E",
    "CPI_New_Vehicles": "CUSR0000SETA01",
    "CPI_Used_Vehicles": "CUSR0000SETA02",
    "CPI_Shelter": "CUSR0000SAH1",
    "CPI_Medical_Care": "CUSR0000SAM",
    "CPI_Transportation_Services": "CUSR0000SAS4",
    "CPI_Airfare": "CUSR0000SETG01",
    "CPI_Services_ex_Energy": "CUSR0000SAS",
    "CPI_Apparel": "CUSR0000SAA",
    "CPI_Recreation": "CUSR0000SAR",
    "CPI_Education_Communication": "CUSR0000SAE1",
}

# ─────────────────────────────────────────────────────────────
# PPI series IDs (PCU = Producer Price by NAICS industry)
# Format: PCU + 6-digit-code + 6-digit-code (primary-product basis)
# ─────────────────────────────────────────────────────────────
PPI_SERIES = {
    "PPI_Agriculture_Crops": "PCU111---111---",
    "PPI_Mining_Oil_Gas": "PCU211---211---",
    "PPI_Mining_Other": "PCU212---212---",
    "PPI_Utilities_Electric": "PCU2211--2211--",
    "PPI_Utilities_Gas": "PCU2212--2212--",
    "PPI_Construction": "PCU236---236---",
    "PPI_Food_Manufacturing": "PCU311---311---",
    "PPI_Petroleum_Coal": "PCU324---324---",
    "PPI_Chemicals": "PCU325---325---",
    "PPI_Plastics_Rubber": "PCU326---326---",
    "PPI_Metals_Primary": "PCU331---331---",
    "PPI_Fabricated_Metal": "PCU332---332---",
    "PPI_Machinery": "PCU333---333---",
    "PPI_Computer_Electronic": "PCU334---334---",
    "PPI_Electrical_Equipment": "PCU335---335---",
    "PPI_Motor_Vehicles": "PCU3361-3361-",
    "PPI_Other_Transportation_Equip": "PCU336---336---",
    "PPI_Wholesale": "PCU42----42----",
    "PPI_Retail": "PCU44-45-44-45-",
    "PPI_Air_Transportation": "PCU481---481---",
    "PPI_Rail_Transportation": "PCU482---482---",
    "PPI_Truck_Transportation": "PCU484---484---",
    "PPI_Water_Transportation": "PCU483---483---",
    "PPI_Warehousing_Storage": "PCU493---493---",
    "PPI_Publishing": "PCU511---511---",
    "PPI_Broadcasting_Telecom": "PCU517---517---",
    "PPI_Finance_Insurance": "PCU52----52----",
    "PPI_Real_Estate": "PCU531---531---",
    "PPI_Professional_Services": "PCU54----54----",
    "PPI_Health_Ambulatory": "PCU621---621---",
    "PPI_Hospitals": "PCU622---622---",
    "PPI_Food_Services": "PCU722---722---",
    "PPI_Accommodation": "PCU721---721---",
}


def bls_request(series_ids: list, start_year: int, end_year: int) -> dict:
    """
    POST request to BLS API v2 for multiple series.
    Returns dict: {series_id: DataFrame with columns [year, period, value]}.
    """
    headers = {"Content-type": "application/json"}
    payload = {
        "seriesid": series_ids,
        "startyear": str(start_year),
        "endyear": str(end_year),
        "calculations": True,
        "annualaverage": True,
    }

    resp = requests.post(BLS_API_V2, data=json.dumps(payload), headers=headers, timeout=60)
    resp.raise_for_status()
    result = resp.json()

    out = {}
    for series in result.get("Results", {}).get("series", []):
        sid = series["seriesID"]
        rows = []
        for obs in series.get("data", []):
            rows.append(
                {
                    "year": int(obs["year"]),
                    "period": obs["period"],
                    "period_name": obs.get("periodName", ""),
                    "value": float(obs["value"]),
                    "footnotes": "; ".join(f.get("text", "") for f in obs.get("footnotes", []) if f),
                }
            )
        out[sid] = pd.DataFrame(rows).sort_values(["year", "period"]).reset_index(drop=True)

    return out


def download_in_batches(
    series_dict: dict, start_year: int, end_year: int, batch_size: int = 25, label: str = ""
) -> pd.DataFrame:
    """
    Download BLS series in batches (API limit: 50 series/request without key,
    500 with key). Returns wide DataFrame indexed by year-period.
    """
    series_ids = list(series_dict.keys())
    names = list(series_dict.values())
    id_to_name = {v: k for k, v in series_dict.items()}

    all_frames = {}
    batches = [names[i: i + batch_size] for i in range(0, len(names), batch_size)]

    for b_idx, batch in enumerate(batches):
        print(f"  {label} — batch {b_idx+1}/{len(batches)} ({len(batch)} series)...")
        try:
            data = bls_request(batch, start_year, end_year)
            for sid, df in data.items():
                col_name = id_to_name.get(sid, sid)
                all_frames[col_name] = df.set_index(["year", "period"])["value"]
        except Exception as e:
            print(f"    Warning: batch failed — {e}")
        if b_idx < len(batches) - 1:
            time.sleep(1.0)  # Rate limiting

    if not all_frames:
        return pd.DataFrame()

    combined = pd.DataFrame(all_frames)
    combined.index.names = ["year", "period"]
    return combined.reset_index()


def download_cpi(start_year: int = 2019, end_year: int = 2022) -> pd.DataFrame:
    """Download CPI series."""
    print("Downloading BLS CPI series...")
    df = download_in_batches(CPI_SERIES, start_year, end_year, label="CPI")
    if not df.empty:
        out_path = RAW_BLS_DIR / "cpi_series.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {df.shape} to {out_path}")
    return df


def download_ppi(start_year: int = 2019, end_year: int = 2022) -> pd.DataFrame:
    """Download PPI series."""
    print("Downloading BLS PPI series...")
    df = download_in_batches(PPI_SERIES, start_year, end_year, label="PPI")
    if not df.empty:
        out_path = RAW_BLS_DIR / "ppi_series.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {df.shape} to {out_path}")
    return df


def compute_log_changes_annual(df: pd.DataFrame, base_year: int, end_year: int) -> pd.DataFrame:
    """
    Compute annual-average log changes between two years.
    Uses period='M13' (annual average) rows.
    """
    annual = df[df["period"] == "M13"].copy()
    base = annual[annual["year"] == base_year].drop(columns=["year", "period"]).set_index([])
    end = annual[annual["year"] == end_year].drop(columns=["year", "period"]).set_index([])

    if base.empty or end.empty:
        return pd.DataFrame()

    base_vals = annual[annual["year"] == base_year].drop(columns=["year", "period"]).values[0]
    end_vals = annual[annual["year"] == end_year].drop(columns=["year", "period"]).values[0]
    cols = [c for c in df.columns if c not in ("year", "period")]

    import numpy as np
    log_changes = pd.Series(
        np.log(end_vals.astype(float)) - np.log(base_vals.astype(float)),
        index=cols,
        name=f"log_change_{base_year}_to_{end_year}",
    )
    return log_changes


def main():
    print("=" * 60)
    print("BLS Data Download")
    print("=" * 60)
    print("Note: BLS API v2 without registration key allows 25 series/request.")
    print("For full download, register at https://data.bls.gov/registrationEngine/")
    print()

    # COVID episode: 2019–2020
    cpi = download_cpi(start_year=2019, end_year=2022)
    time.sleep(2)
    ppi = download_ppi(start_year=2019, end_year=2022)

    print("\nBLS download complete. Files saved to data/raw/bls/")
    print("Next step: run src/io_network/construct_network.py")


if __name__ == "__main__":
    main()
