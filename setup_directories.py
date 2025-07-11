import os

# Defining my folder structure
folders = [
    "data/ffiec",
    "data/macro",
    "src/treasury_forecasting/ingestion",
    "src/treasury_forecasting/modeling",
    "src/treasury_forecasting/dashboard",
    "src/treasury_forecasting/optimization",
    "src/treasury_forecasting/utils",
    "notebooks",
    "dashboards",
    "models",
    "reports"
]

# Create them if they don't exist
for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Folder structure created.")
