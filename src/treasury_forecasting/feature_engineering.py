# src/treasury_forecasting/feature_engineering.py

from pathlib import Path
import pandas as pd
import numpy as np
import os

# Define base directory (assumes script is run from project root)
BASE_DIR = Path(__file__).resolve().parents[2]

# Define input paths
LIQUIDITY_PATH = BASE_DIR / "data" / "cleaned" / "ffiec_liquidity_panel.csv"
MACRO_PATHS = {
    "fed_funds_rate": BASE_DIR / "data" / "macro" / "fed_funds_rate.csv",
    "cpi": BASE_DIR / "data" / "macro" / "cpi.csv",
    "ten_year_treasury": BASE_DIR / "data" / "macro" / "ten_year_treasury.csv"
}

# Define output path
OUTPUT_PATH = BASE_DIR / "data" / "cleaned" / "merged_features.csv"
FDIC_METADATA_PATH = BASE_DIR / "data" / "cleaned" / "fdic_metadata.csv"
BALANCE_PATH = BASE_DIR / "data" / "cleaned" / "ffiec_balance_panel.csv"



def load_and_merge_data() -> pd.DataFrame:
    """
    Loads FFIEC liquidity, balance sheet, and macroeconomic indicators, merges them on date and cert.
    """
    # Load FFIEC liquidity panel
    df_liquidity = pd.read_csv(LIQUIDITY_PATH, parse_dates=["report_date"])
    print(f"Loaded liquidity data: {df_liquidity.shape}")

    # Load FFIEC balance sheet panel
    df_balance = pd.read_csv(BALANCE_PATH, parse_dates=["report_date"])
    print(f"Loaded balance sheet data: {df_balance.shape}")

    # Merge liquidity and balance sheet on report_date and cert
    df_merged = pd.merge(df_liquidity, df_balance, on=["report_date", "cert"], how="left")
    print(f"Merged liquidity + balance: {df_merged.shape}")

    # Merge macro indicators
    for name, path in MACRO_PATHS.items():
        df_macro = pd.read_csv(path, parse_dates=["date"])
        df_macro["report_date"] = df_macro["date"] + pd.offsets.MonthEnd(0)
        df_macro = df_macro.drop(columns=["date"])
        df_merged = pd.merge(df_merged, df_macro, on="report_date", how="left")
        print(f"Merged {name}: {df_macro.shape}")

    print(f"Final merged shape: {df_merged.shape}")
    return df_merged




def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features: ratios and lagged macro variables.
    Cleans missing and infinite values.
    """
    # Create cash to deposit ratio
    df["cash_total"] = df["interest_bearing_cash"] + df["noninterest_cash"]
    df["cash_to_deposit_ratio"] = df["cash_total"] / df["total_deposits"]

    # Create lagged macro variables
    for col in ["fed_funds_rate", "cpi", "ten_year_treasury"]:
        df[f"{col}_lag1"] = df[col].shift(1)

    # Drop rows with missing or infinite values in key columns
    key_cols = ["cash_to_deposit_ratio", "fed_funds_rate_lag1", "cpi_lag1", "ten_year_treasury_lag1"]
    print("\nNull counts before cleaning:")
    print(df[["cash_to_deposit_ratio", "fed_funds_rate_lag1", "cpi_lag1", "ten_year_treasury_lag1"]].isna().sum())
    df_cleaned = df.dropna(subset=key_cols)
    df_cleaned = df_cleaned[np.isfinite(df_cleaned[key_cols]).all(axis=1)]

    print(f"Cleaned data shape: {df_cleaned.shape}")
    return df_cleaned



def merge_fdic_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optionally merges FDIC metadata using the 'cert' field.
    """
    if not FDIC_METADATA_PATH.exists():
        print("FDIC metadata file not found. Skipping merge.")
        return df

    df_fdic = pd.read_csv(FDIC_METADATA_PATH)
    df_fdic = df_fdic.rename(columns={"CERT": "cert"})

    # Merge on 'cert' field (must be present in FFIEC balance sheet data)
    if "cert" not in df.columns:
        print("'cert' field not found in input data. Skipping FDIC merge.")
        return df

    df_merged = pd.merge(df, df_fdic, on="cert", how="left")
    print(f"FDIC metadata merged. New shape: {df_merged.shape}")
    return df_merged




def run_feature_engineering_pipeline() -> None:
    """
    Runs the full feature engineering pipeline and saves the output.
    """
    df_merged = load_and_merge_data()
    df_features = engineer_features(df_merged)
    df_enriched = merge_fdic_metadata(df_features)

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df_features.to_csv(OUTPUT_PATH, index=False)
    print(f"Final dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_feature_engineering_pipeline()