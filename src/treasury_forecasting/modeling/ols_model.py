import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

DATA_PATH = "data/cleaned/merged_features.csv"

def preprocess_features():
    """
    Loads balance sheet-driven features for OLS regression.
    No scaling or log transform applied.
    """
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with shape: {df.shape}")

    target = "cash_to_deposit_ratio"
    features = [
        "interest_bearing_cash",
        "noninterest_cash",
        "total_deposits",
        "total_assets",
        "borrowings"
    ]

    X_raw = df[features].copy()
    y = df[target]

    # Drop rows with missing or infinite values
    X_raw = X_raw.replace([float("inf"), float("-inf")], pd.NA).dropna()
    y = y.loc[X_raw.index]

    print("Selected internal liquidity features")
    return X_raw, y, features


def train_ols_model():
    """
    Trains an OLS model on liquidity-focused predictors.
    """
    X_raw, y, feature_names = preprocess_features()

    X = sm.add_constant(X_raw)
    model = sm.OLS(y, X).fit()

    print("\nLiquidity Model Summary:")
    print(model.summary())

    # Residual plot
    plt.figure(figsize=(8, 5))
    plt.scatter(model.fittedvalues, model.resid, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot: Internal Drivers")
    plt.tight_layout()
    plt.savefig("reports/residual_plot.png")
    print("Residual plot saved to reports/residual_plot.png")


if __name__ == "__main__":
    train_ols_model()

