import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

DATA_PATH = "data/cleaned/merged_features.csv"

def load_data(features, target="cash_to_deposit_ratio"):
    df = pd.read_csv(DATA_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    X = df[features].copy()
    y = df[target].copy()
    return df, X, y

def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def simulate_borrowing_shock(X, shock_percent=1.0):
    shocked_X = X.copy()
    shocked_X["borrowings"] = shocked_X["borrowings"] * (1 + shock_percent)
    return shocked_X

def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print("\n📊 Model Evaluation:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

    # Save residuals plot
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Residuals Distribution (Predicted - Actual)")
    plt.xlabel("Prediction Error")
    plt.tight_layout()
    plt.savefig("reports/residuals_plot.png")
    print("Residuals plot saved to reports/residuals_plot.png")

def plot_simulation_results(y_orig, y_shocked):
    delta = y_shocked - y_orig
    plt.figure(figsize=(8, 5))
    sns.histplot(delta, bins=50, kde=True)
    plt.title("Liquidity Change after Borrowing Shock")
    plt.xlabel("Predicted Liquidity Ratio Δ")
    plt.tight_layout()
    plt.savefig("reports/borrowings_shock_impact.png")
    print("Simulation plot saved to reports/borrowings_shock_impact.png")

    print("\nImpact Summary:")
    print(f"Number of banks with liquidity drop: {(delta < 0).sum()}")
    print(f"Average change in liquidity: {delta.mean():.2f}")
    print(f"Max liquidity drop: {delta.min():.2f}")
    print(f"Max liquidity gain: {delta.max():.2f}")

def flag_at_risk_banks(df, y_shocked, threshold):
    df = df.copy()
    df["liquidity_post_shock"] = y_shocked
    df["risk_flag"] = df["liquidity_post_shock"] < threshold

    num_risky = df["risk_flag"].sum()
    print(f"\nBanks below {threshold}% liquidity: {num_risky}")

    risky_sample = df[df["risk_flag"]].head(10)[
        ["cert", "total_assets", "borrowings", "liquidity_post_shock"]
    ]
    print("\nSample At-Risk Banks:")
    print(risky_sample)

    risky_df = df[df["risk_flag"]].sort_values("liquidity_post_shock").head(20)

    # Plot top 20 riskiest banks
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="liquidity_post_shock",
        y="cert",
        data=risky_df,
        palette="Reds_r"
    )
    plt.title(f"Top 20 Banks Below {threshold}% Liquidity")
    plt.xlabel("Liquidity Ratio after Shock")
    plt.ylabel("Bank Cert Number")
    plt.tight_layout()
    plt.savefig("reports/top_risk_banks.png")
    print("Riskiest banks chart saved to reports/top_risk_banks.png")

    # Save full flagged list
    df.to_csv("reports/flagged_risky_banks.csv", index=False)
    print("Risky banks saved to reports/flagged_risky_banks.csv")

    return df

def main():
    features = [
        "interest_bearing_cash",
        "noninterest_cash",
        "total_deposits",
        "total_assets",
        "borrowings"
    ]

    df, X, y = load_data(features)
    model = train_model(X, y)
    y_pred_orig = model.predict(X)

    evaluate_model(y, y_pred_orig)

    shocked_X = simulate_borrowing_shock(X, shock_percent=1.0)
    y_pred_shocked = model.predict(shocked_X)

    plot_simulation_results(y_pred_orig, y_pred_shocked)

    try:
        threshold = float(input("Enter liquidity threshold (%): "))
    except ValueError:
        threshold = 5.0
        print("Invalid input. Defaulting to 5.0%.")

    flagged_df = flag_at_risk_banks(df, y_pred_shocked, threshold)

if __name__ == "__main__":
    main()

