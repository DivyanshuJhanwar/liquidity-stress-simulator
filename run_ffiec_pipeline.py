from src.treasury_forecasting.ingestion.ffiec_loader import (
    load_ffiec_data,
    extract_liquidity_fields,
    extract_balance_fields,
    save_cleaned_data
)

# -------------------------------
# Part 1: Liquidity Fields
# -------------------------------
ffiec_file_path_1 = "data/ffiec/FFIEC CDR Call Subset of Schedules 2023(1 of 2).txt"
output_path_1 = "data/cleaned/ffiec_liquidity_panel.csv"

df_raw_1 = load_ffiec_data(ffiec_file_path_1)
df_liquidity = extract_liquidity_fields(df_raw_1)
save_cleaned_data(df_liquidity, output_path_1)

# -------------------------------
# Part 2: Balance Sheet Fields
# -------------------------------
ffiec_file_path_2 = "data/ffiec/FFIEC CDR Call Subset of Schedules 2023(2 of 2).txt"
output_path_2 = "data/cleaned/ffiec_balance_panel.csv"

# Load Part 2 with tab delimiter
df_raw_2 = load_ffiec_data(ffiec_file_path_2, sep="\t")

print("\nSearching for 'asset' and 'borrow' in column names:")
for col in df_raw_2.columns:
    if "asset" in col.lower() or "borrow" in col.lower():
        print(col)

# Print all column names to inspect actual field codes
print("\n--- Columns in Part 2 ---")
for i, col in enumerate(df_raw_2.columns):
    print(f"{i}: {col}")

# Temporarily comment out extraction until correct columns are confirmed
df_balance = extract_balance_fields(df_raw_2)
save_cleaned_data(df_balance, output_path_2)
print(df_balance.head())
