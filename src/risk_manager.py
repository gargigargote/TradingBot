from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from src.config import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_RISK_PER_TRADE,
    DEFAULT_KELLY_FRACTION,
    MAX_DAILY_DRAWDOWN_LIMIT,
    logger,
)


class RiskManager:
    """Handles position sizing, fractional Kelly allocation, and drawdown circuit breakers."""

    def __init__(
        self,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
        fractional_kelly: float = DEFAULT_KELLY_FRACTION,
        max_daily_drawdown_limit: float = MAX_DAILY_DRAWDOWN_LIMIT,
        max_single_position_pct: float = 0.25,
        min_shares: int = 1,
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.fractional_kelly = fractional_kelly
        self.max_daily_drawdown_limit = max_daily_drawdown_limit
        self.max_single_position_pct = max_single_position_pct
        self.min_shares = min_shares

        # Day tracking for 3% circuit breaker
        self.current_trading_date = None
        self.daily_start_equity = initial_capital
        self.daily_peak_equity = initial_capital
        self.circuit_breaker_tripped = False
        self.circuit_breaker_history: List[Dict[str, Any]] = []

    def reset_for_new_day_if_needed(self, current_timestamp: Any, current_equity: float):
        """Rolls daily benchmark over when trading date changes."""
        if current_timestamp is None:
            return

        ts_date = pd.to_datetime(current_timestamp).date()
        if self.current_trading_date is None or ts_date != self.current_trading_date:
            if self.circuit_breaker_tripped:
                logger.info(f"New day ({ts_date}): resetting intraday circuit breaker.")
            self.current_trading_date = ts_date
            self.daily_start_equity = current_equity
            self.daily_peak_equity = current_equity
            self.circuit_breaker_tripped = False

    def check_circuit_breaker(self, current_timestamp: Any, current_equity: float) -> Tuple[bool, float]:
        """Checks if intraday drawdown exceeds 3% from day's start or peak."""
        self.reset_for_new_day_if_needed(current_timestamp, current_equity)

        if current_equity > self.daily_peak_equity:
            self.daily_peak_equity = current_equity

        dd_from_start = (self.daily_start_equity - current_equity) / (self.daily_start_equity + 1e-8)
        dd_from_peak = (self.daily_peak_equity - current_equity) / (self.daily_peak_equity + 1e-8)
        current_dd = max(dd_from_start, dd_from_peak)

        if current_dd >= self.max_daily_drawdown_limit:
            if not self.circuit_breaker_tripped:
                self.circuit_breaker_tripped = True
                self.circuit_breaker_history.append(
                    {
                        "timestamp": current_timestamp,
                        "date": str(self.current_trading_date),
                        "daily_start_equity": self.daily_start_equity,
                        "current_equity": current_equity,
                        "drawdown_pct": current_dd,
                    }
                )
                logger.warning(
                    f"Circuit breaker tripped on {self.current_trading_date}: "
                    f"drawdown {current_dd * 100:.2f}% >= {self.max_daily_drawdown_limit * 100:.2f}%. Trading paused."
                )
            return True, current_dd

        return False, current_dd

    @staticmethod
    def calculate_kelly_fraction(
        win_rate: float,
        win_loss_ratio: float,
        kelly_fraction_multiplier: float = 0.5,
    ) -> float:
        """Full Kelly: f* = p - (1 - p) / b. Scaled by multiplier (default 0.5 for half-Kelly)."""
        if win_loss_ratio <= 0 or win_rate <= 0:
            return 0.0

        q = 1.0 - win_rate
        full_kelly = win_rate - (q / win_loss_ratio)
        if full_kelly <= 0:
            return 0.0

        scaled = full_kelly * max(0.0, min(1.0, kelly_fraction_multiplier))
        return min(scaled, 1.0)

    def calculate_position_size(
        self,
        current_equity: float,
        entry_price: float,
        stop_loss_price: float,
        atr: Optional[float] = None,
        win_rate: Optional[float] = None,
        win_loss_ratio: Optional[float] = None,
    ) -> int:
        """Sizes trade based on fixed risk amount / stop loss distance."""
        if current_equity <= 0 or entry_price <= 0:
            return 0

        # Don't take new positions if circuit breaker is active
        if self.circuit_breaker_tripped:
            return 0

        # Stop distance with ATR floor
        stop_distance = abs(entry_price - stop_loss_price)
        min_distance = (atr * 0.5) if (atr and atr > 0) else (entry_price * 0.005)
        stop_distance = max(stop_distance, min_distance)

        # Dollar risk: equity * risk_pct
        risk_budget = current_equity * self.risk_per_trade
        base_shares = int(risk_budget / stop_distance)

        # Scale by Kelly if historical trade stats exist
        if win_rate is not None and win_loss_ratio is not None and win_rate > 0 and win_loss_ratio > 0:
            k_pct = self.calculate_kelly_fraction(
                win_rate=win_rate,
                win_loss_ratio=win_loss_ratio,
                kelly_fraction_multiplier=self.fractional_kelly,
            )
            if k_pct > 0:
                max_kelly_shares = int((current_equity * k_pct) / entry_price)
                shares = min(base_shares, max_kelly_shares) if max_kelly_shares > 0 else base_shares
            else:
                shares = max(int(base_shares * 0.5), self.min_shares)
        else:
            shares = base_shares

        # Never allocate more than max_single_position_pct of portfolio to one ticker
        cap_shares = int((current_equity * self.max_single_position_pct) / entry_price)
        return max(min(shares, cap_shares), 0)
