import numpy as np
import pandas as pd


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Rolling simple moving average."""
    if window <= 0:
        raise ValueError("window must be > 0")
    return series.rolling(window=window, min_periods=window).mean()


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with alpha = 2 / (span + 1)."""
    if span <= 0:
        raise ValueError("span must be > 0")
    return series.ewm(span=span, adjust=False).mean()


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    use_wilder: bool = True,
) -> pd.Series:
    """Average True Range.
    
    True Range = max(H - L, |H - C_prev|, |L - C_prev|)
    Default uses Wilder's exponential smoothing (alpha = 1 / window).
    """
    if window <= 0:
        raise ValueError("window must be > 0")

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]

    if use_wilder:
        return tr.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    return tr.rolling(window=window, min_periods=window).mean()


def calculate_z_score(
    series: pd.Series,
    window: int = 20,
    ddof: int = 0,
    eps: float = 1e-8,
) -> pd.Series:
    """Z-Score relative to rolling moving average and rolling standard deviation.
    
    Z = (P - SMA) / (std + eps)
    """
    if window <= 1:
        raise ValueError("window must be > 1")

    sma = calculate_sma(series, window=window)
    rolling_std = series.rolling(window=window, min_periods=window).std(ddof=ddof)
    # eps prevents division by zero during flat market bars
    return (series - sma) / (rolling_std + eps)


def calculate_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
    ddof: int = 0,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Computes Bollinger Bands: middle, upper, lower, bandwidth, and %B."""
    if window <= 1:
        raise ValueError("window must be > 1")

    middle = calculate_sma(series, window=window)
    rolling_std = series.rolling(window=window, min_periods=window).std(ddof=ddof)

    upper = middle + (num_std * rolling_std)
    lower = middle - (num_std * rolling_std)
    bandwidth = (upper - lower) / (middle + eps)
    pct_b = (series - lower) / ((upper - lower) + eps)

    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_pct_b": pct_b,
        },
        index=series.index,
    )


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches an OHLCV frame with technical indicators."""
    out = df.copy()
    close, high, low = out["Close"], out["High"], out["Low"]

    # Trend indicators
    out["sma_20"] = calculate_sma(close, window=20)
    out["sma_50"] = calculate_sma(close, window=50)
    out["ema_50"] = calculate_ema(close, span=50)
    out["ema_200"] = calculate_ema(close, span=200)

    # Volatility / ATR
    out["atr_14"] = calculate_atr(high, low, close, window=14)
    out["atr_50"] = calculate_atr(high, low, close, window=50)

    # 20-day Z-score
    out["z_score_20"] = calculate_z_score(close, window=20)

    # Bollinger Bands
    bands = calculate_bollinger_bands(close, window=20, num_std=2.0)
    for col in bands.columns:
        out[col] = bands[col]

    return out
