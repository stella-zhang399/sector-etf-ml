from pathlib import Path

import pandas as pd


# -----------------------------
# File paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "macro_daily.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "macro_features.csv"
)


# -----------------------------
# Load raw macro data
# -----------------------------

def load_macro_data():
    df = pd.read_csv(
        RAW_DATA_PATH,
        parse_dates=["DATE"],
    )

    df = (
        df.sort_values("DATE")
        .reset_index(drop=True)
    )

    return df


# -----------------------------
# Convert to monthly data
# -----------------------------

def build_monthly_macro(df):

    monthly = (
        df.set_index("DATE")
        .resample("ME")
        .last()
        .reset_index()
    )

    monthly = monthly.rename(
        columns={"DATE": "Date"}
    )

    return monthly


# -----------------------------
# Engineer macro features
# -----------------------------

def add_macro_features(monthly):

    df = monthly.copy()

    # Yield curve spread
    df["yield_spread_10y_2y"] = (
        df["DGS10"] - df["DGS2"]
    )

    # Monthly changes
    change_columns = [
        "DGS10",
        "DGS2",
        "DFF",
        "VIXCLS",
        "yield_spread_10y_2y",
    ]

    for column in change_columns:
        df[f"{column}_change_1m"] = (
            df[column].diff()
        )

    return df


# -----------------------------
# Main pipeline
# -----------------------------

def main():

    macro_daily = load_macro_data()

    print("Raw macro data:")
    print(macro_daily.shape)

    monthly = build_monthly_macro(macro_daily)

    macro_features = add_macro_features(monthly)

    macro_features.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("Monthly macro dataset:")
    print(macro_features.shape)

    print()
    print("Date range:")
    print(
        macro_features["Date"].min(),
        "to",
        macro_features["Date"].max(),
    )

    print()
    print("Columns:")
    print(macro_features.columns.tolist())

    print()
    print("Missing values:")
    print(macro_features.isna().sum())

    print()
    print(macro_features.head())

    print()
    print(f"Saved macro features to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()