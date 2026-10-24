from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.features import compute_all_features


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class MarketRegime(str, Enum):
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    HIGH_VOL_CHOP = "HIGH_VOL_CHOP"
    LOW_VOL_MEAN_REVERT = "LOW_VOL_MEAN_REVERT"


@dataclass
class TradingSignal:
    timestamp: Any
    symbol: str
    signal_type: SignalType
    price: float
    z_score: float
    regime: MarketRegime
    stop_loss: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, symbol: str = "ASSET") -> pd.DataFrame:
        pass

    @abstractmethod
    def on_bar(self, current_bar: pd.Series, symbol: str = "ASSET") -> TradingSignal:
        pass


class DualLayerRegimeBollingerStrategy(BaseStrategy):
    """Dual-layer trend-filtered mean-reversion strategy:
    
    1. Regime Filter:
       - Uses EMA 50 vs EMA 200 to establish macro trend direction.
       - Uses ATR 14 vs ATR 50 ratio to avoid trading into volatility explosions.
       
    2. Execution Trigger:
       - Long entry when price extends below lower Bollinger Band (Z <= -2.0).
       - Exit when price reverts to 20-day mean (Z >= 0.0) or hits trailing ATR stop.
    """

    def __init__(
        self,
        z_entry_threshold: float = -2.0,
        z_exit_threshold: float = 0.0,
        atr_stop_multiplier: float = 2.0,
        allow_bear_entries: bool = False,
        vol_expansion_threshold: float = 1.25,
    ):
        super().__init__(name="DualLayerRegimeBollinger")
        self.z_entry_threshold = z_entry_threshold
        self.z_exit_threshold = z_exit_threshold
        self.atr_stop_multiplier = atr_stop_multiplier
        self.allow_bear_entries = allow_bear_entries
        self.vol_expansion_threshold = vol_expansion_threshold

        # Runtime position tracking
        self.in_position = False
        self.entry_price = 0.0
        self.trailing_stop = 0.0

    def classify_regime(
        self,
        ema_50: float,
        ema_200: float,
        atr_14: float,
        atr_50: float,
    ) -> MarketRegime:
        """Classify current market structure into trend vs chop."""
        if pd.isna(ema_50) or pd.isna(ema_200) or pd.isna(atr_14):
            return MarketRegime.LOW_VOL_MEAN_REVERT

        is_bullish = ema_50 >= ema_200
        vol_ratio = (atr_14 / atr_50) if (not pd.isna(atr_50) and atr_50 > 0) else 1.0

        if vol_ratio > self.vol_expansion_threshold:
            return MarketRegime.HIGH_VOL_CHOP
        elif is_bullish:
            return MarketRegime.BULL_TREND
        elif not is_bullish and vol_ratio <= 1.0:
            return MarketRegime.LOW_VOL_MEAN_REVERT
        else:
            return MarketRegime.BEAR_TREND

    def reset_state(self):
        self.in_position = False
        self.entry_price = 0.0
        self.trailing_stop = 0.0

    def on_bar(self, current_bar: pd.Series, symbol: str = "ASSET") -> TradingSignal:
        price = float(current_bar["Close"])
        timestamp = current_bar.name if hasattr(current_bar, "name") else pd.Timestamp.now()

        z_score = float(current_bar.get("z_score_20", 0.0))
        atr = float(current_bar.get("atr_14", price * 0.01))
        ema_50 = float(current_bar.get("ema_50", np.nan))
        ema_200 = float(current_bar.get("ema_200", np.nan))
        atr_50 = float(current_bar.get("atr_50", np.nan))

        regime = self.classify_regime(ema_50, ema_200, atr, atr_50)
        favorable_regime = (regime in (MarketRegime.BULL_TREND, MarketRegime.LOW_VOL_MEAN_REVERT)) or self.allow_bear_entries

        if not self.in_position:
            # Check for oversold pullback entry
            if favorable_regime and (z_score <= self.z_entry_threshold) and not pd.isna(z_score):
                self.in_position = True
                self.entry_price = price
                stop_dist = max(atr * self.atr_stop_multiplier, price * 0.005)
                self.trailing_stop = price - stop_dist

                return TradingSignal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=price,
                    z_score=z_score,
                    regime=regime,
                    stop_loss=self.trailing_stop,
                    metadata={"reason": "Oversold Z-Score Entry", "atr": atr},
                )
            return TradingSignal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.HOLD,
                price=price,
                z_score=z_score,
                regime=regime,
            )

        # In position: update trailing stop ratcheting up
        stop_dist = max(atr * self.atr_stop_multiplier, price * 0.005)
        new_stop = price - stop_dist
        if new_stop > self.trailing_stop:
            self.trailing_stop = new_stop

        # Exit on mean reversion target or trailing stop breach
        hit_target = (z_score >= self.z_exit_threshold) and not pd.isna(z_score)
        hit_stop = price <= self.trailing_stop

        if hit_target or hit_stop:
            reason = "Mean Reversion Target (Z >= 0.0)" if hit_target else "Trailing ATR Stop Triggered"
            self.in_position = False
            prev_entry = self.entry_price
            self.entry_price = 0.0

            return TradingSignal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=price,
                z_score=z_score,
                regime=regime,
                stop_loss=self.trailing_stop,
                metadata={"reason": reason, "entry_price": prev_entry, "atr": atr},
            )

        return TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=SignalType.HOLD,
            price=price,
            z_score=z_score,
            regime=regime,
            stop_loss=self.trailing_stop,
            metadata={"entry_price": self.entry_price},
        )

    def generate_signals(self, df: pd.DataFrame, symbol: str = "ASSET") -> pd.DataFrame:
        self.reset_state()
        data = compute_all_features(df)

        signals: List[str] = []
        regimes: List[str] = []
        stops: List[float] = []
        reasons: List[str] = []

        for idx in range(len(data)):
            bar = data.iloc[idx]
            sig = self.on_bar(bar, symbol=symbol)
            signals.append(sig.signal_type.value)
            regimes.append(sig.regime.value)
            stops.append(sig.stop_loss)
            reasons.append(sig.metadata.get("reason", ""))

        data["signal"] = signals
        data["regime"] = regimes
        data["stop_loss"] = stops
        data["signal_reason"] = reasons
        return data
