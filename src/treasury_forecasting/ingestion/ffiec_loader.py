import pandas as pd
import os


def load_ffiec_data(file_path: str, sep: str = "\t") -> pd.DataFrame:
    """
    Load the FFIEC raw data using the specified delimiter and return as a DataFrame.
    Includes basic validation logs.
    """
    df = pd.read_csv(file_path, sep=sep, low_memory=False)
    print(f"Loaded FFIEC data with shape: {df.shape}")

    # Basic diagnostics
    print("\nColumn preview:")
    print(df.columns[:10].tolist())

    print("\nNull counts (top 5):")
    print(df.isnull().sum().sort_values(ascending=False).head())

    print("\nSample rows:")
    print(df.head(2))

    return df


def extract_liquidity_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts liquidity-related fields from FFIEC Part 1.
    """
    required_columns = {
        "Reporting Period End Date": "report_date",
        "FDIC Certificate Number": "cert",  # Added for merging
        "Financial Institution Name": "institution",
        "RCON2200": "total_deposits",
        "RCON0081": "interest_bearing_cash",
        "RCON0071": "noninterest_cash"
    }

    df_subset = df[list(required_columns.keys())].rename(columns=required_columns)
    df_subset["report_date"] = pd.to_datetime(df_subset["report_date"])
    df_subset["cert"] = pd.to_numeric(df_subset["cert"], errors="coerce")

    for col in ["interest_bearing_cash", "noninterest_cash", "total_deposits"]:
        df_subset[col] = pd.to_numeric(df_subset[col], errors="coerce")

    print(f"Extracted liquidity fields with shape: {df_subset.shape}")
    return df_subset


def extract_balance_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts balance sheet fields from FFIEC Part 2.
    """
    required_columns = {
        "Reporting Period End Date": "report_date",
        "FDIC Certificate Number": "cert",
        "RIAD0093": "total_assets",
        "RIAD3196": "borrowings"
    }

    df_subset = df[list(required_columns.keys())].rename(columns=required_columns)
    df_subset["report_date"] = pd.to_datetime(df_subset["report_date"])
    df_subset["cert"] = pd.to_numeric(df_subset["cert"], errors="coerce")

    for col in ["total_assets", "borrowings"]:
        df_subset[col] = pd.to_numeric(df_subset[col], errors="coerce")

    print(f"Extracted balance sheet fields with shape: {df_subset.shape}")
    return df_subset


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the cleaned DataFrame to a CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
