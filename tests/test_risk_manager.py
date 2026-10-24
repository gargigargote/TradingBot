"""Unit tests for Risk Management Engine: Sizing, Fractional Kelly, and Circuit Breaker."""

import pytest
import pandas as pd
from src.risk_manager import RiskManager


def test_dynamic_position_sizing_formula():
    """Verify Shares = (Capital * Risk%) / (Stop Loss Distance)."""
    rm = RiskManager(
        initial_capital=100000.0,
        risk_per_trade=0.02,  # $2,000 risk
        max_single_position_pct=1.0,  # Uncapped for formula test
    )
    
    # Capital = $100,000, Risk = $2,000
    # Entry = 100.0, Stop = 95.0, Distance = 5.0
    # Expected shares = 2000 / 5 = 400 shares
    shares = rm.calculate_position_size(
        current_equity=100000.0,
        entry_price=100.0,
        stop_loss_price=95.0,
        atr=2.5,
    )
    assert shares == 400


def test_fractional_kelly_criterion():
    """Verify Kelly Criterion calculations for positive and negative edge."""
    # 60% win rate, 2.0 win/loss ratio
    # f* = 0.6 - (0.4 / 2.0) = 0.40
    # Half-Kelly (0.5x) = 0.20 (20%)
    f_half = RiskManager.calculate_kelly_fraction(
        win_rate=0.6,
        win_loss_ratio=2.0,
        kelly_fraction_multiplier=0.5,
    )
    assert f_half == pytest.approx(0.20)
    
    # Negative expectation (win rate 30%, win/loss 1.0)
    # f* = 0.3 - 0.7 = -0.4 -> should clamp to 0.0
    f_neg = RiskManager.calculate_kelly_fraction(
        win_rate=0.3,
        win_loss_ratio=1.0,
        kelly_fraction_multiplier=0.5,
    )
    assert f_neg == 0.0


def test_circuit_breaker_halts_trading_on_3pct_drawdown():
    """Verify 3% daily drawdown triggers circuit breaker and pauses allocation."""
    rm = RiskManager(
        initial_capital=100000.0,
        max_daily_drawdown_limit=0.03,  # 3% limit
    )
    
    day1 = pd.Timestamp("2023-05-01 09:30:00")
    
    # Starting equity $100,000 -> Active, no trip
    is_tripped, dd = rm.check_circuit_breaker(day1, current_equity=100000.0)
    assert not is_tripped
    assert dd == pytest.approx(0.0)
    
    # Drop to $98,000 (2% loss) -> still under 3% limit
    is_tripped, dd = rm.check_circuit_breaker(day1, current_equity=98000.0)
    assert not is_tripped
    assert dd == pytest.approx(0.02)
    
    # Drop to $96,500 (3.5% loss >= 3%) -> Circuit breaker trips!
    is_tripped, dd = rm.check_circuit_breaker(day1, current_equity=96500.0)
    assert is_tripped
    assert dd >= 0.03
    assert rm.circuit_breaker_tripped
    
    # Sizing request during active circuit breaker must return 0 shares
    shares = rm.calculate_position_size(
        current_equity=96500.0,
        entry_price=100.0,
        stop_loss_price=95.0,
    )
    assert shares == 0
    
    # Next trading day arrives -> Circuit breaker resets
    day2 = pd.Timestamp("2023-05-02 09:30:00")
    is_tripped_day2, dd2 = rm.check_circuit_breaker(day2, current_equity=96500.0)
    assert not is_tripped_day2
    assert not rm.circuit_breaker_tripped
    assert rm.daily_start_equity == 96500.0
