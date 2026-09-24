from pathlib import Path

import numpy as np
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

BASELINE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_model_dataset.csv"
)

ADVANCED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "advanced_model_dataset.csv"
)


# -----------------------------
# Load data
# -----------------------------

def load_data():

    panel = pd.read_csv(
        MODEL_PANEL_PATH,
        parse_dates=["Date"],
    )

    baseline = pd.read_csv(
        BASELINE_DATA_PATH,
        parse_dates=["Date"],
    )

    return panel, baseline


# -----------------------------
# Rolling beta and correlation
# -----------------------------

def add_rolling_market_features(panel):

    ticker_frames = []

    for ticker, group in panel.groupby(
        "Ticker",
        sort=False,
    ):

        group = (
            group
            .sort_values("Date")
            .copy()
            .reset_index(drop=True)
        )

        for window in [6, 12]:

            rolling_cov = (
                group["monthly_return"]
                .rolling(
                    window=window,
                    min_periods=window,
                )
                .cov(group["spy_monthly_return"])
            )

            rolling_spy_var = (
                group["spy_monthly_return"]
                .rolling(
                    window=window,
                    min_periods=window,
                )
                .var()
            )

            group[f"beta_{window}m"] = (
                rolling_cov
                / rolling_spy_var
            )

            group[f"corr_spy_{window}m"] = (
                group["monthly_return"]
                .rolling(
                    window=window,
                    min_periods=window,
                )
                .corr(
                    group["spy_monthly_return"]
                )
            )

        ticker_frames.append(group)

    result = pd.concat(
        ticker_frames,
        ignore_index=True,
    )

    return result


# -----------------------------
# Cross-sectional features
# -----------------------------

def add_cross_sectional_features(df):

    df = df.copy()

    rank_columns = [
        "mom_3m",
        "mom_6m",
        "mom_12m",
        "vol_3m",
        "vol_6m",
        "vol_12m",
    ]

    for column in rank_columns:

        df[f"{column}_rank_pct"] = (
            df.groupby("Date")[column]
            .rank(
                pct=True,
                method="average",
            )
        )

    # Cross-sectional dispersion in
    # current monthly sector returns
    df["sector_return_dispersion"] = (
        df.groupby("Date")["monthly_return"]
        .transform("std")
    )

    return df


# -----------------------------
# Build advanced dataset
# -----------------------------

def build_advanced_dataset(
    panel,
    baseline,
):

    advanced = (
        add_rolling_market_features(panel)
        .pipe(add_cross_sectional_features)
    )

    advanced_features = [
        "beta_6m",
        "beta_12m",
        "corr_spy_6m",
        "corr_spy_12m",

        "mom_3m_rank_pct",
        "mom_6m_rank_pct",
        "mom_12m_rank_pct",

        "vol_3m_rank_pct",
        "vol_6m_rank_pct",
        "vol_12m_rank_pct",

        "sector_return_dispersion",
    ]

    advanced_subset = advanced[
        ["Date", "Ticker"]
        + advanced_features
    ].copy()

    final = baseline.merge(
        advanced_subset,
        on=["Date", "Ticker"],
        how="left",
        validate="one_to_one",
    )

    final = final.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    final = (
        final
        .dropna(subset=advanced_features)
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    return final, advanced_features


# -----------------------------
# Main
# -----------------------------

def main():

    panel, baseline = load_data()

    final, advanced_features = (
        build_advanced_dataset(
            panel,
            baseline,
        )
    )

    print("Baseline dataset:")
    print(baseline.shape)

    print()
    print("Advanced dataset:")
    print(final.shape)

    print()
    print("Date range:")
    print(
        final["Date"].min(),
        "to",
        final["Date"].max(),
    )

    print()
    print("New features:")
    print(advanced_features)

    print()
    print("Missing values in new features:")
    print(
        final[advanced_features]
        .isna()
        .sum()
    )

    print()
    print("Advanced feature summary:")
    print(
        final[advanced_features]
        .describe()
        .T
    )

    final.to_csv(
        ADVANCED_DATA_PATH,
        index=False,
    )

    print()
    print("Saved advanced dataset to:")
    print(ADVANCED_DATA_PATH)


if __name__ == "__main__":
    main()