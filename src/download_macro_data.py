from pathlib import Path

import pandas as pd
from pandas_datareader import data as pdr


# -----------------------------
# Configuration
# -----------------------------

FRED_SERIES = [
    # Existing
    "DGS10",
    "DGS2",
    "DFF",
    "VIXCLS",

    # New
    "BAA10Y",       # Baa corporate credit spread vs. 10Y Treasury
    "DCOILWTICO",   # WTI crude oil price
    "CPIAUCSL",     # Consumer Price Index
    "UNRATE",       # Unemployment rate
]

START_DATE = "2013-01-01"
END_DATE = "2025-12-31"


# -----------------------------
# File paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RAW_DATA_DIR / "macro_daily.csv"


# -----------------------------
# Download FRED data
# -----------------------------

def download_macro_data():

    print("Downloading FRED data...")

    macro = pdr.DataReader(
        FRED_SERIES,
        "fred",
        START_DATE,
        END_DATE,
    )

    macro = macro.reset_index()

    print()
    print("Downloaded data:")
    print(macro.shape)

    print()
    print(macro.head())

    print()
    print("Missing values:")
    print(macro.isna().sum())

    macro.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(f"Saved raw macro data to:")
    print(OUTPUT_PATH)

    return macro


if __name__ == "__main__":
    download_macro_data()