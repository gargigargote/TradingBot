"""Unit tests for quantitative technical features and mathematical indicators."""

import pytest
import numpy as np
import pandas as pd

from src.features import (
    calculate_sma,
    calculate_ema,
    calculate_atr,
    calculate_z_score,
    calculate_bollinger_bands,
    compute_all_features,
)


def test_calculate_sma():
    """Verify Simple Moving Average calculation and edge cases."""
    prices = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    sma = calculate_sma(prices, window=3)
    
    # First 2 values should be NaN
    assert pd.isna(sma.iloc[0])
    assert pd.isna(sma.iloc[1])
    assert sma.iloc[2] == pytest.approx(20.0)
    assert sma.iloc[3] == pytest.approx(30.0)
    assert sma.iloc[4] == pytest.approx(40.0)

    with pytest.raises(ValueError):
        calculate_sma(prices, window=0)


def test_calculate_ema():
    """Verify Exponential Moving Average calculation."""
    prices = pd.Series([100.0, 102.0, 104.0, 103.0, 105.0])
    ema = calculate_ema(prices, span=3)
    
    assert len(ema) == len(prices)
    assert ema.iloc[0] == pytest.approx(100.0)
    # alpha = 2 / (3 + 1) = 0.5
    # ema[1] = 0.5 * 102 + 0.5 * 100 = 101.0
    assert ema.iloc[1] == pytest.approx(101.0)
    # ema[2] = 0.5 * 104 + 0.5 * 101 = 102.5
    assert ema.iloc[2] == pytest.approx(102.5)


def test_calculate_atr():
    """Verify Average True Range calculation with gap up and gap down."""
    high = pd.Series([105.0, 112.0, 108.0, 115.0])
    low = pd.Series([98.0, 104.0, 101.0, 107.0])
    close = pd.Series([102.0, 110.0, 103.0, 114.0])
    
    atr = calculate_atr(high, low, close, window=2, use_wilder=False)
    
    assert len(atr) == 4
    # TR0 = High0 - Low0 = 105 - 98 = 7.0
    # TR1 = max(112-104=8, |112-102|=10, |104-102|=2) = 10.0
    # Rolling 2-period ATR at idx 1 = (7 + 10) / 2 = 8.5
    assert atr.iloc[1] == pytest.approx(8.5)


def test_calculate_z_score():
    """Verify Z-Score calculation relative to 20-day rolling mean and standard deviation."""
    # Constant series should have z-score ~ 0 without ZeroDivisionError
    flat = pd.Series([100.0] * 30)
    z_flat = calculate_z_score(flat, window=20)
    assert np.all(np.abs(z_flat.dropna()) < 1e-4)

    # Standardized movement
    data = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0] * 6)
    z = calculate_z_score(data, window=20)
    assert not z.dropna().empty
    assert -5.0 < z.dropna().mean() < 5.0


def test_calculate_bollinger_bands():
    """Verify Bollinger Bands bounds: Upper > Middle > Lower, and Bandwidth, %B."""
    prices = pd.Series([100.0 + i * 0.5 for i in range(50)])
    bb = calculate_bollinger_bands(prices, window=20, num_std=2.0)
    
    valid = bb.dropna()
    assert (valid["bb_upper"] > valid["bb_middle"]).all()
    assert (valid["bb_middle"] > valid["bb_lower"]).all()
    assert (valid["bb_bandwidth"] > 0).all()


def test_compute_all_features():
    """Verify comprehensive technical features enrichment."""
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(60)],
            "High": [102.0 + i for i in range(60)],
            "Low": [99.0 + i for i in range(60)],
            "Close": [101.0 + i for i in range(60)],
            "Volume": [100000] * 60,
        },
        index=dates,
    )
    
    enriched = compute_all_features(df)
    
    expected_cols = [
        "sma_20", "sma_50", "ema_50", "ema_200",
        "atr_14", "atr_50", "z_score_20",
        "bb_middle", "bb_upper", "bb_lower", "bb_bandwidth", "bb_pct_b"
    ]
    for col in expected_cols:
        assert col in enriched.columns, f"Missing feature column: {col}"
