from pathlib import Path

import pandas as pd


# -----------------------------
# File paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_features.csv"
)

MODEL_PANEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_panel.csv"
)


# -----------------------------
# Load feature data
# -----------------------------

def load_market_features():
    df = pd.read_csv(
        FEATURE_DATA_PATH,
        parse_dates=["Date"],
    )

    df = (
        df.sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    return df


# -----------------------------
# Construct sector + SPY panel
# -----------------------------

def build_model_panel(df):

    # Separate SPY from sector ETFs
    spy = (
        df[df["Ticker"] == "SPY"]
        .copy()
        .sort_values("Date")
    )

    sectors = (
        df[df["Ticker"] != "SPY"]
        .copy()
        .sort_values(["Ticker", "Date"])
    )

    # ---------------------------------
    # Create next-month returns
    # ---------------------------------

    sectors["next_etf_return"] = (
        sectors.groupby("Ticker")["monthly_return"]
        .shift(-1)
    )

    spy["next_spy_return"] = (
        spy["monthly_return"]
        .shift(-1)
    )

    # ---------------------------------
    # Rename SPY features
    # ---------------------------------

    spy = spy.rename(
        columns={
            "monthly_return": "spy_monthly_return",
            "mom_1m": "spy_mom_1m",
            "mom_3m": "spy_mom_3m",
            "mom_6m": "spy_mom_6m",
            "mom_12m": "spy_mom_12m",
            "vol_3m": "spy_vol_3m",
            "vol_6m": "spy_vol_6m",
            "vol_12m": "spy_vol_12m",
            "drawdown_12m": "spy_drawdown_12m",
        }
    )

    spy_columns = [
        "Date",
        "spy_monthly_return",
        "spy_mom_1m",
        "spy_mom_3m",
        "spy_mom_6m",
        "spy_mom_12m",
        "spy_vol_3m",
        "spy_vol_6m",
        "spy_vol_12m",
        "spy_drawdown_12m",
        "next_spy_return",
    ]

    spy = spy[spy_columns]

    # ---------------------------------
    # Merge sector and SPY features
    # ---------------------------------

    panel = sectors.merge(
        spy,
        on="Date",
        how="left",
    )

    # ---------------------------------
    # Relative features
    # ---------------------------------

    for window in [1, 3, 6, 12]:
        panel[f"relative_mom_{window}m"] = (
            panel[f"mom_{window}m"]
            - panel[f"spy_mom_{window}m"]
        )

    for window in [3, 6, 12]:
        panel[f"relative_vol_{window}m"] = (
            panel[f"vol_{window}m"]
            - panel[f"spy_vol_{window}m"]
        )

    panel["relative_drawdown_12m"] = (
        panel["drawdown_12m"]
        - panel["spy_drawdown_12m"]
    )

    # ---------------------------------
    # Binary target
    # ---------------------------------

    panel["target"] = (
        panel["next_etf_return"]
        > panel["next_spy_return"]
    ).astype("Int64")

    # Target is undefined when next-month
    # returns are unavailable
    missing_future = (
        panel["next_etf_return"].isna()
        | panel["next_spy_return"].isna()
    )

    panel.loc[missing_future, "target"] = pd.NA

    return panel


# -----------------------------
# Main pipeline
# -----------------------------

def main():

    market_features = load_market_features()

    panel = build_model_panel(market_features)

    panel.to_csv(
        MODEL_PANEL_PATH,
        index=False,
    )

    print("Model panel shape:")
    print(panel.shape)

    print()
    print("Date range:")
    print(panel["Date"].min(), "to", panel["Date"].max())

    print()
    print("Observations by ETF:")
    print(panel["Ticker"].value_counts().sort_index())

    print()
    print("Target distribution:")
    print(panel["target"].value_counts(dropna=False))

    print()
    print("Missing values:")
    print(panel.isna().sum())

    print()
    print(f"Saved model panel to:")
    print(MODEL_PANEL_PATH)


if __name__ == "__main__":
    main()