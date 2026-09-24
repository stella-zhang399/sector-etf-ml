from pathlib import Path

import pandas as pd
import yfinance as yf


# -----------------------------
# Configuration
# -----------------------------

TICKERS = [
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Health Care
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLI",   # Industrials
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLRE",  # Real Estate
    "SPY",   # S&P 500 benchmark
]

START_DATE = "2015-01-01"
END_DATE = "2026-01-01"


# -----------------------------
# File paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Download data
# -----------------------------

def download_market_data():
    all_data = []

    for ticker in TICKERS:
        print(f"Downloading {ticker}...")

        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            print(f"Warning: no data returned for {ticker}")
            continue

        df = df.reset_index()

        # yfinance can sometimes return MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["Ticker"] = ticker

        all_data.append(df)

    market_data = pd.concat(all_data, ignore_index=True)

    output_path = RAW_DATA_DIR / "market_daily.csv"
    market_data.to_csv(output_path, index=False)

    print()
    print(f"Saved {len(market_data):,} rows to:")
    print(output_path)

    return market_data


if __name__ == "__main__":
    download_market_data()