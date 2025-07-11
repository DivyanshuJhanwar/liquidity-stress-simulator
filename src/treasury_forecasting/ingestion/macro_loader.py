# src/treasury_forecasting/ingestion/macro_loader.py

from fredapi import Fred
import pandas as pd
import requests
import os


def fetch_fred_data(series_id: str, api_key: str, start_date: str = "2000-01-01") -> pd.DataFrame:
    """
    Fetches time series data from the FRED API and returns a DataFrame.
    Includes basic validation logs.
    """
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date
    }

    response = requests.get(url, params=params)
    data = response.json()

    observations = data.get("observations", [])
    df = pd.DataFrame(observations)

    if df.empty:
        print(f"No data returned for {series_id}")
        return df

    df = df[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])

    print(f"\nFRED series '{series_id}' loaded with shape: {df.shape}")
    print(df.head(2))

    return df


def fetch_and_save_fred_series(series_map: dict, api_key: str, output_dir: str) -> None:
    """
    Fetches multiple FRED series and saves each to a CSV file.
    """
    for name, series_id in series_map.items():
        df = fetch_fred_data(series_id=series_id, api_key=api_key)
        if not df.empty:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved {name} to {output_path}")


def fetch_fdic_metadata(limit: int = 1000) -> pd.DataFrame:
    """
    Fetches basic bank metadata from the FDIC BankFind API.
    """
    url = "https://banks.data.fdic.gov/api/institutions"
    params = {
        "filters": "",  # loosen filter
        "fields": "NAME,CERT,ASSET,CHARTERTYPE",
        "limit": limit,
        "format": "json"
    }

    response = requests.get(url, params=params)
    records = response.json().get("data", [])

    if not records:
        print("No records returned from FDIC API.")
        return pd.DataFrame()

    df = pd.json_normalize([r["data"] for r in records])
    print(f"Retrieved {len(df)} bank records from FDIC.")
    return df

