from pathlib import Path

import pandas as pd


# -----------------------------
# File paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "market_daily.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MONTHLY_DATA_PATH = PROCESSED_DATA_DIR / "market_monthly.csv"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load raw data
# -----------------------------

def load_market_data():
    df = pd.read_csv(
        RAW_DATA_PATH,
        parse_dates=["Date"],
    )

    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

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
        .pct_change()
    )

    return monthly


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
    print()
    print(market_monthly.head())

    market_monthly.to_csv(MONTHLY_DATA_PATH, index=False)

    print()
    print(f"Saved monthly data to:")
    print(MONTHLY_DATA_PATH)


if __name__ == "__main__":
    main()