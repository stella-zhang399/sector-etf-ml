from pathlib import Path

import pandas as pd


# -----------------------------
# File paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "advanced_model_dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "regime_model_dataset.csv"
)


# -----------------------------
# Load data
# -----------------------------

def load_data():

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["Date"],
    )

    return df


# -----------------------------
# Regime indicators
# -----------------------------

def add_regime_features(df):

    df = df.copy()

    # -----------------------------
    # Market regimes
    # -----------------------------

    # Elevated market volatility
    df["high_vix_regime"] = (
        df["VIXCLS"] >= 20
    ).astype(int)

    # Inverted 10Y-2Y yield curve
    df["inverted_yield_curve"] = (
        df["yield_spread_10y_2y"] < 0
    ).astype(int)

    # Rising long-term interest rates
    df["rising_rate_regime"] = (
        df["DGS10_change_1m"] > 0
    ).astype(int)

    # Widening corporate credit spreads
    df["credit_stress_regime"] = (
        df["BAA10Y_change_1m"] > 0
    ).astype(int)

    # Rising inflation
    df["rising_inflation_regime"] = (
        df["CPI_YOY_change_1m_lag1"] > 0
    ).astype(int)

    # Rising unemployment
    df["rising_unemployment_regime"] = (
        df["UNRATE_change_1m_lag1"] > 0
    ).astype(int)

    # -----------------------------
    # Regime interactions
    # -----------------------------

    # Does momentum behave differently
    # during high-volatility markets?
    df["momentum_x_high_vix"] = (
        df["mom_12m_rank_pct"]
        * df["high_vix_regime"]
    )

    # Does market beta matter differently
    # when volatility is elevated?
    df["beta_x_high_vix"] = (
        df["beta_12m"]
        * df["high_vix_regime"]
    )

    # Does sector momentum behave differently
    # when the yield curve is inverted?
    df["momentum_x_inverted_curve"] = (
        df["mom_12m_rank_pct"]
        * df["inverted_yield_curve"]
    )

    # Does sector momentum behave differently
    # when credit conditions deteriorate?
    df["momentum_x_credit_stress"] = (
        df["mom_12m_rank_pct"]
        * df["credit_stress_regime"]
    )

    return df


# -----------------------------
# Main
# -----------------------------

def main():

    df = load_data()

    regime_df = add_regime_features(df)

    regime_features = [
        "high_vix_regime",
        "inverted_yield_curve",
        "rising_rate_regime",
        "credit_stress_regime",
        "rising_inflation_regime",
        "rising_unemployment_regime",
        "momentum_x_high_vix",
        "beta_x_high_vix",
        "momentum_x_inverted_curve",
        "momentum_x_credit_stress",
    ]

    print("Input dataset:")
    print(df.shape)

    print()
    print("Regime dataset:")
    print(regime_df.shape)

    print()
    print("Regime frequencies:")
    print(
        regime_df[
            regime_features[:6]
        ].mean()
    )

    print()
    print("Missing values:")
    print(
        regime_df[
            regime_features
        ].isna().sum()
    )

    regime_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("Saved regime dataset to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()