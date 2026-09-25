from pathlib import Path

import pandas as pd


# -----------------------------
# File paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PANEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_panel.csv"
)

MACRO_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "macro_features.csv"
)

FINAL_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_model_dataset.csv"
)


# -----------------------------
# Load datasets
# -----------------------------

def load_data():

    panel = pd.read_csv(
        MODEL_PANEL_PATH,
        parse_dates=["Date"],
    )

    macro = pd.read_csv(
        MACRO_FEATURES_PATH,
        parse_dates=["Date"],
    )

    return panel, macro


# -----------------------------
# Merge market + macro data
# -----------------------------

def merge_datasets(panel, macro):

    merged = panel.merge(
        macro,
        on="Date",
        how="left",
        validate="many_to_one",
    )

    return merged


# -----------------------------
# Create analysis-ready sample
# -----------------------------

def create_final_sample(df):

    # Predictor columns that require complete observations
    required_features = [
        # ETF features
        "mom_1m",
        "mom_3m",
        "mom_6m",
        "mom_12m",
        "vol_3m",
        "vol_6m",
        "vol_12m",
        "drawdown_12m",

        # SPY features
        "spy_mom_1m",
        "spy_mom_3m",
        "spy_mom_6m",
        "spy_mom_12m",
        "spy_vol_3m",
        "spy_vol_6m",
        "spy_vol_12m",
        "spy_drawdown_12m",

        # Relative ETF vs. SPY features
        "relative_mom_1m",
        "relative_mom_3m",
        "relative_mom_6m",
        "relative_mom_12m",
        "relative_vol_3m",
        "relative_vol_6m",
        "relative_vol_12m",
        "relative_drawdown_12m",

        # Macro levels
        "DGS10",
        "DGS2",
        "DFF",
        "VIXCLS",
        "yield_spread_10y_2y",

        # Macro monthly changes
        "DGS10_change_1m",
        "DGS2_change_1m",
        "DFF_change_1m",
        "VIXCLS_change_1m",
        "yield_spread_10y_2y_change_1m",

        # Additional macro features
        "BAA10Y",
        "BAA10Y_change_1m",
        "DCOILWTICO",
        "DCOILWTICO_change_1m",
        "CPI_YOY_lag1",
        "CPI_YOY_change_1m_lag1",
        "UNRATE_lag1",
        "UNRATE_change_1m_lag1",
    ]

    required_columns = required_features + [
        "target",
        "next_excess_return",
    ]

    final = (
        df.dropna(subset=required_columns)
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    final["target"] = final["target"].astype(int)

    if final.duplicated(subset=["Date", "Ticker"]).any():
        raise ValueError("Duplicate ETF-month observations found.")

    return final


# -----------------------------
# Main pipeline
# -----------------------------

def main():

    panel, macro = load_data()

    print("ETF model panel:")
    print(panel.shape)

    print()
    print("Macro dataset:")
    print(macro.shape)

    merged = merge_datasets(panel, macro)

    print()
    print("Merged dataset:")
    print(merged.shape)

    print()
    print("Missing macro values after merge:")
    macro_columns = macro.columns.drop("Date")
    print(merged[macro_columns].isna().sum())

    final = create_final_sample(merged)

    print()
    print("Final analysis-ready dataset:")
    print(final.shape)

    print()
    print("Date range:")
    print(
        final["Date"].min(),
        "to",
        final["Date"].max(),
    )

    print()
    print("Observations by ETF:")
    print(
        final["Ticker"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Target distribution:")
    print(
        final["target"]
        .value_counts(normalize=False)
        .sort_index()
    )

    print()
    print("Remaining missing values:")
    print(final.isna().sum())

    final.to_csv(
        FINAL_DATA_PATH,
        index=False,
    )

    print()
    print(f"Saved final dataset to:")
    print(FINAL_DATA_PATH)


if __name__ == "__main__":
    main()