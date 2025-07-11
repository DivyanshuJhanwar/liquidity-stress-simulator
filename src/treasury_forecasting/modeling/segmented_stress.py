import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RISKY_DATA_PATH = "reports/flagged_risky_banks.csv"
FDIC_METADATA_PATH = "data/cleaned/fdic_metadata.csv" 

def load_merged_data():
    # Load your FDIC metadata
    fdic_df = pd.read_csv(FDIC_METADATA_PATH)

    # Rename columns for consistency
    fdic_df = fdic_df.rename(columns={
        "CERT": "cert",
        "NAME": "charter_class",     # Using bank name as a placeholder
        "ASSET": "total_assets"
    })

    # Create asset tier bands (assuming assets are in millions)
    bins = [0, 250, 500, 1000, float("inf")]
    labels = ["Under $250M", "$250M–$500M", "$500M–$1B", "Over $1B"]
    fdic_df["asset_tier"] = pd.cut(fdic_df["total_assets"], bins=bins, labels=labels)

    # Load risky bank list and merge
    risky_df = pd.read_csv(RISKY_DATA_PATH)
    merged = pd.merge(
        risky_df,
        fdic_df[["cert", "charter_class", "asset_tier"]],
        on="cert",
        how="left"
    )

    print(f"Merged dataset with shape: {merged.shape}")
    return merged

def plot_by_charter_class(df):
    charter_order = df["charter_class"].value_counts().index.tolist()

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x="charter_class",
        y="liquidity_post_shock",
        data=df,
        order=charter_order,
        palette="coolwarm"
    )
    plt.title("Liquidity Distribution by Charter Type")
    plt.ylabel("Liquidity Ratio after Shock")
    plt.xlabel("Charter Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/charter_liquidity_boxplot.png")
    print("Charter plot saved to reports/charter_liquidity_boxplot.png")

def plot_by_asset_tier(df):
    tier_order = df["asset_tier"].value_counts().index.tolist()

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x="asset_tier",
        y="liquidity_post_shock",
        data=df,
        order=tier_order,
        palette="viridis"
    )
    plt.title("Liquidity Distribution by Asset Tier")
    plt.ylabel("Liquidity Ratio after Shock")
    plt.xlabel("Asset Tier")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/asset_tier_liquidity_boxplot.png")
    print("Asset tier plot saved to reports/asset_tier_liquidity_boxplot.png")

def summarize_risk_by_group(df):
    grouped = df.groupby(["charter_class", "asset_tier"])
    summary = grouped["liquidity_post_shock"].agg(["count", "mean", "min"]).reset_index()
    summary.to_csv("reports/segmented_liquidity_summary.csv", index=False)
    print("Segment summary saved to reports/segmented_liquidity_summary.csv")

def main():
    df = load_merged_data()
    plot_by_charter_class(df)
    plot_by_asset_tier(df)
    summarize_risk_by_group(df)

if __name__ == "__main__":
    main()
