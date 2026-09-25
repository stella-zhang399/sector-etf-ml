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

    # -----------------------------
    # Existing macro features
    # -----------------------------

    df["yield_spread_10y_2y"] = (
        df["DGS10"] - df["DGS2"]
    )

    change_columns = [
        "DGS10",
        "DGS2",
        "DFF",
        "VIXCLS",
        "BAA10Y",
        "DCOILWTICO",
        "yield_spread_10y_2y",
    ]

    for column in change_columns:
        df[f"{column}_change_1m"] = (
            df[column].diff()
        )

    # -----------------------------
    # Inflation
    # -----------------------------

    # Year-over-year CPI inflation
    cpi_yoy = (
        df["CPIAUCSL"]
        .pct_change(periods=12)
        * 100
    )

    # CPI for month t is generally not known
    # by the end of month t, so lag it one month.
    df["CPI_YOY_lag1"] = (
        cpi_yoy.shift(1)
    )

    df["CPI_YOY_change_1m_lag1"] = (
        df["CPI_YOY_lag1"].diff()
    )

    # -----------------------------
    # Labor market
    # -----------------------------

    # Same idea: use the previous month's
    # unemployment reading.
    df["UNRATE_lag1"] = (
        df["UNRATE"].shift(1)
    )

    df["UNRATE_change_1m_lag1"] = (
        df["UNRATE_lag1"].diff()
    )

    # -----------------------------
    # Keep only leakage-safe features
    # -----------------------------

    feature_columns = [
        "Date",

        # Existing levels
        "DGS10",
        "DGS2",
        "DFF",
        "VIXCLS",
        "yield_spread_10y_2y",

        # Existing changes
        "DGS10_change_1m",
        "DGS2_change_1m",
        "DFF_change_1m",
        "VIXCLS_change_1m",
        "yield_spread_10y_2y_change_1m",

        # Credit conditions
        "BAA10Y",
        "BAA10Y_change_1m",

        # Oil
        "DCOILWTICO",
        "DCOILWTICO_change_1m",

        # Inflation — publication lag respected
        "CPI_YOY_lag1",
        "CPI_YOY_change_1m_lag1",

        # Labor market — publication lag respected
        "UNRATE_lag1",
        "UNRATE_change_1m_lag1",
    ]

    return df[feature_columns]


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