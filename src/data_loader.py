from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf

from src.config import DATA_DIR, USE_OFFLINE_DATA_FALLBACK, logger


def generate_synthetic_ohlcv(
    symbol: str = "SYNTH",
    start: str = "2022-01-01",
    end: str = "2024-01-01",
    interval: str = "1d",
    initial_price: float = 150.0,
    daily_drift: float = 0.0004,
    daily_vol: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic OHLCV data using a geometric random walk with jumps."""
    rng = np.random.default_rng(seed)
    date_range = pd.date_range(start=start, end=end, freq="B")
    n = len(date_range)
    if n == 0:
        date_range = pd.date_range(start=start, periods=252, freq="B")
        n = len(date_range)

    # Daily log returns + occasional jump regime
    returns = rng.normal(loc=daily_drift, scale=daily_vol, size=n)
    jumps = rng.choice([0.0, 0.02, -0.02], size=n, p=[0.90, 0.05, 0.05])
    returns += jumps

    close = initial_price * np.cumprod(np.exp(returns))

    # Synthetic Open, High, Low
    prev_close = np.roll(close, 1)
    prev_close[0] = initial_price
    open_prices = prev_close * (1 + rng.normal(0, daily_vol * 0.3, size=n))

    spread = np.abs(rng.normal(daily_vol * 1.5, daily_vol * 0.5, size=n))
    high = np.maximum(open_prices, close) * (1 + spread * 0.6) + 0.01
    low = np.maximum(np.minimum(open_prices, close) * (1 - spread * 0.6) - 0.01, 0.01)

    volume = (1_000_000 * rng.lognormal(0, 0.4, size=n) * (1 + np.abs(returns) / daily_vol * 0.5)).astype(int)

    df = pd.DataFrame(
        {
            "Open": np.round(open_prices, 2),
            "High": np.round(high, 2),
            "Low": np.round(low, 2),
            "Close": np.round(close, 2),
            "Volume": volume,
        },
        index=date_range,
    )
    df.index.name = "Date"
    return df


class DataLoader:
    def __init__(self, cache_dir: Path = DATA_DIR, offline_fallback: bool = USE_OFFLINE_DATA_FALLBACK):
        self.cache_dir = cache_dir
        self.offline_fallback = offline_fallback
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, start: str, end: str, interval: str) -> Path:
        clean = symbol.replace("/", "_").replace("^", "").upper()
        return self.cache_dir / f"{clean}_{start}_{end}_{interval}.csv"

    def fetch_historical_data(
        self,
        symbol: str = "SPY",
        start: str = "2022-01-01",
        end: str = "2024-01-01",
        interval: str = "1d",
        force_reload: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        cache_file = self._cache_path(symbol, start, end, interval)

        # 1. Use disk cache if present
        if not force_reload and cache_file.exists():
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if not df.empty and all(c in df.columns for c in ["Open", "High", "Low", "Close", "Volume"]):
                    return df
            except Exception:
                pass

        # 2. Offline synthetic mode
        if offline:
            df = generate_synthetic_ohlcv(symbol=symbol, start=start, end=end, interval=interval)
            df.to_csv(cache_file)
            return df

        # 3. Fetch from Yahoo Finance
        try:
            logger.info(f"Fetching {symbol} from Yahoo Finance ({start} to {end})...")
            raw = yf.download(
                tickers=symbol,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            if raw.empty or len(raw) < 5:
                raise ValueError("insufficient data returned")

            raw = raw.rename(columns={c: c.capitalize() for c in raw.columns})
            required = ["Open", "High", "Low", "Close", "Volume"]
            df = raw[required].dropna()
            df.index.name = "Date"

            df.to_csv(cache_file)
            return df

        except Exception as err:
            logger.warning(f"Failed to fetch {symbol} online: {err}")
            if self.offline_fallback:
                logger.info("Falling back to synthetic data generator.")
                df = generate_synthetic_ohlcv(symbol=symbol, start=start, end=end, interval=interval)
                df.to_csv(cache_file)
                return df
            raise
