"""Unit tests for Dual-Layer Regime Detection and Bollinger Z-Score Strategy."""

import pytest
import pandas as pd
import numpy as np

from src.strategy import (
    DualLayerRegimeBollingerStrategy,
    MarketRegime,
    SignalType,
    TradingSignal,
)


def test_market_regime_classification():
    """Verify Layer 1 market regime detection using EMA 50/200 and ATR."""
    strat = DualLayerRegimeBollingerStrategy()
    
    # 1. Bull trend: EMA 50 > EMA 200, moderate ATR
    regime = strat.classify_regime(ema_50=150.0, ema_200=140.0, atr_14=2.0, atr_50=2.0)
    assert regime == MarketRegime.BULL_TREND
    
    # 2. Bear trend: EMA 50 < EMA 200, elevated ATR
    regime = strat.classify_regime(ema_50=130.0, ema_200=140.0, atr_14=2.4, atr_50=2.0)
    assert regime == MarketRegime.BEAR_TREND
    
    # 3. High volatility chop: ATR 14 expanding > 1.25x ATR 50
    regime = strat.classify_regime(ema_50=150.0, ema_200=140.0, atr_14=3.5, atr_50=2.0)
    assert regime == MarketRegime.HIGH_VOL_CHOP


def test_bollinger_z_score_entry_and_exit_signals():
    """Verify Layer 2 entry (Z <= -2.0) and exit (Z >= 0.0)."""
    strat = DualLayerRegimeBollingerStrategy(z_entry_threshold=-2.0, z_exit_threshold=0.0)
    
    # Bar 1: Normal bar, Z = -0.5 -> HOLD
    bar1 = pd.Series({"Close": 100.0, "z_score_20": -0.5, "atr_14": 2.0, "ema_50": 105.0, "ema_200": 95.0, "atr_50": 2.0})
    sig1 = strat.on_bar(bar1, symbol="TEST")
    assert sig1.signal_type == SignalType.HOLD
    assert not strat.in_position
    
    # Bar 2: Oversold pullback, Z = -2.2 in Bull Trend -> BUY
    bar2 = pd.Series({"Close": 95.0, "z_score_20": -2.2, "atr_14": 2.0, "ema_50": 105.0, "ema_200": 95.0, "atr_50": 2.0})
    sig2 = strat.on_bar(bar2, symbol="TEST")
    assert sig2.signal_type == SignalType.BUY
    assert strat.in_position
    assert sig2.stop_loss == pytest.approx(95.0 - (2.0 * 2.0))  # 91.0
    
    # Bar 3: Price recovers partially, Z = -1.0 -> HOLD (trailing stop maintained)
    bar3 = pd.Series({"Close": 98.0, "z_score_20": -1.0, "atr_14": 2.0, "ema_50": 105.0, "ema_200": 95.0, "atr_50": 2.0})
    sig3 = strat.on_bar(bar3, symbol="TEST")
    assert sig3.signal_type == SignalType.HOLD
    assert strat.in_position
    
    # Bar 4: Price reverts to 20-day mean, Z = +0.2 -> SELL (Target Reached)
    bar4 = pd.Series({"Close": 103.0, "z_score_20": 0.2, "atr_14": 2.0, "ema_50": 105.0, "ema_200": 95.0, "atr_50": 2.0})
    sig4 = strat.on_bar(bar4, symbol="TEST")
    assert sig4.signal_type == SignalType.SELL
    assert not strat.in_position
    assert "Mean Reversion" in sig4.metadata.get("reason", "")


def test_trailing_stop_exit():
    """Verify trailing stop ratchets up and triggers protective exit on drawdown."""
    strat = DualLayerRegimeBollingerStrategy(atr_stop_multiplier=2.0)
    
    # Enter position at 100 with ATR = 2.0 (initial stop = 96.0)
    bar_entry = pd.Series({"Close": 100.0, "z_score_20": -2.5, "atr_14": 2.0, "ema_50": 105.0, "ema_200": 95.0, "atr_50": 2.0})
    sig_entry = strat.on_bar(bar_entry)
    assert sig_entry.signal_type == SignalType.BUY
    assert strat.trailing_stop == pytest.approx(96.0)
    
    # Price rises to 110, ATR = 2.0 -> trailing stop ratchets to 110 - 4 = 106.0
    bar_up = pd.Series({"Close": 110.0, "z_score_20": -0.5, "atr_14": 2.0, "ema_50": 105.0, "ema_200": 95.0, "atr_50": 2.0})
    strat.on_bar(bar_up)
    assert strat.trailing_stop == pytest.approx(106.0)
    
    # Price falls to 105.0 (< 106.0 trailing stop) -> triggers SELL
    bar_stop = pd.Series({"Close": 105.0, "z_score_20": -1.2, "atr_14": 2.0, "ema_50": 105.0, "ema_200": 95.0, "atr_50": 2.0})
    sig_stop = strat.on_bar(bar_stop)
    assert sig_stop.signal_type == SignalType.SELL
    assert "Trailing ATR Stop" in sig_stop.metadata.get("reason", "")
    assert not strat.in_position
