from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# File paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "market_daily.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MONTHLY_DATA_PATH = PROCESSED_DATA_DIR / "market_monthly.csv"
FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "market_features.csv"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load raw data
# -----------------------------

def load_market_data():
    df = pd.read_csv(
        RAW_DATA_PATH,
        parse_dates=["Date"],
    )

    df = (
        df.sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    return df


# -----------------------------
# Convert daily prices to monthly
# -----------------------------

def build_monthly_data(df):
    monthly = (
        df.set_index("Date")
        .groupby("Ticker")["Adj Close"]
        .resample("ME")
        .last()
        .rename("adj_close")
        .reset_index()
    )

    monthly["monthly_return"] = (
        monthly.groupby("Ticker")["adj_close"]
        .pct_change(fill_method=None)
    )

    return monthly


# -----------------------------
# Feature engineering
# -----------------------------

def add_market_features(monthly):
    df = monthly.copy()

    df = (
        df.sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    # 1-month momentum
    df["mom_1m"] = (
        df.groupby("Ticker")["adj_close"]
        .pct_change(periods=1, fill_method=None)
    )

    # Multi-month momentum
    for window in [3, 6, 12]:
        df[f"mom_{window}m"] = (
            df.groupby("Ticker")["adj_close"]
            .pct_change(periods=window, fill_method=None)
        )

    # Rolling annualized volatility
    for window in [3, 6, 12]:
        df[f"vol_{window}m"] = (
            df.groupby("Ticker")["monthly_return"]
            .transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=window,
                ).std() * np.sqrt(12)
            )
        )

    # 12-month drawdown from rolling high
    rolling_high = (
        df.groupby("Ticker")["adj_close"]
        .transform(
            lambda x: x.rolling(
                window=12,
                min_periods=12,
            ).max()
        )
    )

    df["drawdown_12m"] = (
        df["adj_close"] / rolling_high - 1
    )

    return df


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    market_daily = load_market_data()

    print("Raw data:")
    print(market_daily.shape)

    market_monthly = build_monthly_data(market_daily)

    print()
    print("Monthly data:")
    print(market_monthly.shape)

    market_monthly.to_csv(
        MONTHLY_DATA_PATH,
        index=False,
    )

    market_features = add_market_features(market_monthly)

    market_features.to_csv(
        FEATURE_DATA_PATH,
        index=False,
    )

    print()
    print("Feature dataset:")
    print(market_features.shape)

    print()
    print("Columns:")
    print(market_features.columns.tolist())

    print()
    print("Missing values:")
    print(market_features.isna().sum())

    print()
    print(f"Saved feature data to:")
    print(FEATURE_DATA_PATH)


if __name__ == "__main__":
    main()