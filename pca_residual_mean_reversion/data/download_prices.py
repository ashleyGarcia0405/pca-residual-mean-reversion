from __future__ import annotations

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pathlib import Path


START_DATE = "2011-01-01"
END_DATE = None             # None = up to latest available
N_STOCKS = 200

RAW_DIR = Path("../../data/raw")
PROCESSED_DIR = Path("../../data/processed")
INTERIM_DIR = Path("../../data/interim")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


def get_sp500_tickers() -> list[str]:
    """
    Pull current S&P 500 constituents from Wikipedia.

    Note:
    This gives current constituents, not historical constituents.
    This creates survivorship bias.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    table = pd.read_html(resp.text)[0]

    tickers = table["Symbol"].tolist()

    # Yahoo uses '-' instead of '.' for tickers like BRK.B and BF.B
    tickers = [ticker.replace(".", "-") for ticker in tickers]

    return tickers


def download_yfinance_data(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download daily adjusted close and volume data.
    """
    data = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        actions=False,
        group_by="column",
        threads=True,
        progress=True,
    )

    adj_close = data["Adj Close"].copy()
    volume = data["Volume"].copy()

    # Drop columns that are completely empty
    adj_close = adj_close.dropna(axis=1, how="all")
    volume = volume[adj_close.columns]

    return adj_close, volume


def select_liquid_universe(
    adj_close: pd.DataFrame,
    volume: pd.DataFrame,
    n_stocks: int = 200,
    lookback_days: int = 252,
    max_missing_fraction: float = 0.05,
) -> list[str]:
    """
    Select liquid stocks using average dollar volume.

    Dollar volume = adjusted close * share volume.
    """
    missing_fraction = adj_close.isna().mean()
    eligible = missing_fraction[missing_fraction <= max_missing_fraction].index

    adj_close = adj_close[eligible]
    volume = volume[eligible]

    dollar_volume = adj_close * volume

    avg_dollar_volume = dollar_volume.tail(lookback_days).mean()
    liquid_names = (
        avg_dollar_volume
        .sort_values(ascending=False)
        .head(n_stocks)
        .index
        .tolist()
    )

    return liquid_names


def clean_prices(adj_close: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Keep selected names and clean small missing gaps.
    Forward-fill small gaps, then drop any remaining incomplete rows.
    """
    prices = adj_close[tickers].copy()
    prices = prices.ffill(limit=5)
    prices = prices.dropna(axis=0, how="any")

    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def winsorize_returns(
    log_returns: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    q_lo = log_returns.quantile(lower)
    q_hi = log_returns.quantile(upper)
    return log_returns.clip(lower=q_lo, upper=q_hi, axis=1)


def compute_missing_summary(adj_close_raw: pd.DataFrame) -> pd.DataFrame:
    n = len(adj_close_raw)
    missing_count = adj_close_raw.isna().sum()
    missing_frac = missing_count / n
    return pd.DataFrame({
        "total_observations": n,
        "missing_count": missing_count,
        "missing_fraction": missing_frac,
    }).sort_values("missing_fraction", ascending=False)


def main() -> None:
    tickers = get_sp500_tickers()
    print(f"Pulled {len(tickers)} S&P 500 tickers.")

    adj_close, volume = download_yfinance_data(tickers)

    adj_close.to_parquet(RAW_DIR / "sp500_adj_close_raw.parquet")
    volume.to_parquet(RAW_DIR / "sp500_volume_raw.parquet")

    missing_summary = compute_missing_summary(adj_close)
    missing_summary.to_csv(INTERIM_DIR / "missing_data_summary.csv")

    liquid_tickers = select_liquid_universe(
        adj_close=adj_close,
        volume=volume,
        n_stocks=N_STOCKS,
    )

    prices = clean_prices(adj_close, liquid_tickers)
    log_returns = compute_log_returns(prices)
    log_returns_win = winsorize_returns(log_returns)

    prices.to_parquet(PROCESSED_DIR / "sp500_liquid_adj_close.parquet")
    prices.to_csv(PROCESSED_DIR / "prices.csv")
    log_returns_win.to_parquet(PROCESSED_DIR / "log_returns.parquet")
    log_returns_win.to_csv(PROCESSED_DIR / "log_returns.csv")
    pd.Series(liquid_tickers, name="ticker").to_csv(
        PROCESSED_DIR / "ticker_universe.csv",
        index=False,
    )

    print(f"Final universe size: {prices.shape[1]} stocks")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"Return observations: {log_returns_win.shape[0]} days x {log_returns_win.shape[1]} stocks")
    print(f"Saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()