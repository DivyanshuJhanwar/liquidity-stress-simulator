import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/cleaned/merged_features.csv"

def load_and_prepare_data():
    """
    Load dataset and select liquidity-focused features.
    """
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with shape: {df.shape}")

    features = [
        "interest_bearing_cash",
        "noninterest_cash",
        "total_deposits",
        "total_assets",
        "borrowings"
    ]
    target = "cash_to_deposit_ratio"

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    X = df[features]
    y = df[target]

    return X, y, features


def train_random_forest_model():
    """
    Trains a Random Forest regression model.
    """
    X, y, feature_names = load_and_prepare_data()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)

    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = mean_squared_error(y, predictions, squared=False)

    print(f"\nModel Performance:")
    print(f"R² Score        : {r2:.4f}")
    print(f"MAE             : {mae:.2f}")
    print(f"RMSE            : {rmse:.2f}")

    # Feature importance plot
    importances = model.feature_importances_
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=feature_names)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("reports/tree_feature_importance.png")
    print("Feature importance plot saved to reports/tree_feature_importance.png")

    return model


if __name__ == "__main__":
    train_random_forest_model()
