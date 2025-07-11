# run_macro_ingestion.py

from src.treasury_forecasting.ingestion.macro_loader import fetch_fdic_metadata, fetch_and_save_fred_series

# Replace with your actual FRED API key
FRED_API_KEY = "cdbb8e31a622e9582785c38aa41f33fb"
OUTPUT_DIR = "data/macro/"

FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "ten_year_treasury": "GS10"
}

# Fetch and save macroeconomic indicators
fetch_and_save_fred_series(series_map=FRED_SERIES, api_key=FRED_API_KEY, output_dir=OUTPUT_DIR)

# Fetch FDIC metadata and save to CSV
df_fdic = fetch_fdic_metadata()
df_fdic.to_csv("data/cleaned/fdic_metadata.csv", index=False)
