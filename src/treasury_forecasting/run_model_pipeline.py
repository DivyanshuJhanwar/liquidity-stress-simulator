# run_model_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

DATA_PATH = "data/cleaned/merged_features.csv"
THRESHOLD = 5.0
SHOCK_PERCENT = 1.0

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

def simulate_borrowing_shock(X, percent):
    shocked_X = X.copy()
    shocked_X["borrowings"] *= (1 + percent)
    return shocked_X

def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print("📊 Model Evaluation")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Residuals Distribution (Predicted - Actual)")
    plt.xlabel("Prediction Error")
    plt.tight_layout()
    plt.savefig("reports/residuals_plot.png")

def plot_simulation_results(y_orig, y_shocked):
    delta = y_shocked - y_orig
    plt.figure(figsize=(8, 5))
    sns.histplot(delta, bins=50, kde=True)
    plt.title("Liquidity Change after Borrowing Shock")
    plt.xlabel("Predicted Liquidity Ratio Δ")
    plt.tight_layout()
    plt.savefig("reports/borrowings_shock_impact.png")

def flag_at_risk_banks(df, y_shocked, threshold):
    df["liquidity_post_shock"] = y_shocked
    df["risk_flag"] = df["liquidity_post_shock"] < threshold

    risky_df = df[df["risk_flag"]].sort_values("liquidity_post_shock").head(20)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="liquidity_post_shock", y="cert", data=risky_df, palette="Reds_r")
    plt.title(f"Top 20 Banks Below {threshold}% Liquidity")
    plt.xlabel("Liquidity Ratio after Shock")
    plt.ylabel("Bank Cert Number")
    plt.tight_layout()
    plt.savefig("reports/top_risk_banks.png")

    df.to_csv("reports/flagged_risky_banks.csv", index=False)
    print(f"✅ Pipeline completed. Flagged data saved to reports/flagged_risky_banks.csv")

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

    shocked_X = simulate_borrowing_shock(X, SHOCK_PERCENT)
    y_pred_shocked = model.predict(shocked_X)
    plot_simulation_results(y_pred_orig, y_pred_shocked)

    flag_at_risk_banks(df, y_pred_shocked, THRESHOLD)

if __name__ == "__main__":
    main()
