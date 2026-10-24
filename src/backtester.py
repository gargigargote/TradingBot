import argparse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
import numpy as np
import pandas as pd

from src.audit_logger import log_trades_bulk
from src.config import (
    DEFAULT_COMMISSION_RATE,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_SLIPPAGE_RATE,
    SQLITE_DB_PATH,
    logger,
)
from src.data_loader import DataLoader
from src.features import compute_all_features
from src.risk_manager import RiskManager
from src.strategy import (
    BaseStrategy,
    DualLayerRegimeBollingerStrategy,
    SignalType,
    TradingSignal,
)


class EventType(str, Enum):
    BAR = "BAR"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


@dataclass
class BarEvent:
    timestamp: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    data: Dict[str, Any] = field(default_factory=dict)
    event_type: EventType = EventType.BAR


@dataclass
class OrderEvent:
    timestamp: pd.Timestamp
    symbol: str
    order_type: str
    direction: str  # BUY / SELL
    quantity: int
    target_price: float
    stop_loss: float = 0.0
    event_type: EventType = EventType.ORDER


@dataclass
class FillEvent:
    timestamp: pd.Timestamp
    symbol: str
    direction: str
    quantity: int
    fill_price: float
    commission: float
    slippage: float
    net_cost: float
    event_type: EventType = EventType.FILL


@dataclass
class TradeRecord:
    session_id: str
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    quantity: int
    entry_price: float
    exit_price: float
    slippage: float
    commission: float
    gross_pnl: float
    net_pnl: float
    return_pct: float
    exit_reason: str


class EventDrivenBacktester:
    """Event-driven simulator that walks historical bars sequentially,
    models transaction slippage and commission friction, and tracks cash & equity.
    """

    def __init__(
        self,
        symbol: str = "SPY",
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
        commission_rate: float = DEFAULT_COMMISSION_RATE,
        strategy: Optional[BaseStrategy] = None,
        risk_manager: Optional[RiskManager] = None,
        audit_to_db: bool = True,
        db_path: Path = SQLITE_DB_PATH,
    ):
        self.symbol = symbol.upper()
        self.initial_capital = initial_capital
        self.slippage_rate = slippage_rate
        self.commission_rate = commission_rate
        self.audit_to_db = audit_to_db
        self.db_path = db_path

        self.strategy = strategy or DualLayerRegimeBollingerStrategy()
        self.risk_manager = risk_manager or RiskManager(initial_capital=initial_capital)

        # Simulator state
        self.cash = initial_capital
        self.position_qty = 0
        self.avg_entry_price = 0.0
        self.entry_timestamp: Optional[pd.Timestamp] = None
        self.session_id = f"bt_{uuid.uuid4().hex[:8]}"

        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[TradeRecord] = []
        self.fill_events: List[FillEvent] = []

    def reset(self):
        self.cash = self.initial_capital
        self.position_qty = 0
        self.avg_entry_price = 0.0
        self.entry_timestamp = None
        self.equity_curve.clear()
        self.trades.clear()
        self.fill_events.clear()
        self.strategy.reset_state()
        self.session_id = f"bt_{uuid.uuid4().hex[:8]}"

    def execute_order(self, order: OrderEvent) -> FillEvent:
        price = order.target_price
        qty = order.quantity

        if order.direction == "BUY":
            slippage_amt = price * self.slippage_rate
            fill_price = price + slippage_amt
            comm = fill_price * qty * self.commission_rate
            total_slippage = slippage_amt * qty
            total_cost = (fill_price * qty) + comm
            self.cash -= total_cost

            old_qty = self.position_qty
            old_cost = old_qty * self.avg_entry_price
            self.position_qty += qty
            self.avg_entry_price = (old_cost + (fill_price * qty)) / self.position_qty
            self.entry_timestamp = order.timestamp
            net_cost = total_cost

        else:  # SELL
            slippage_amt = price * self.slippage_rate
            fill_price = price - slippage_amt
            comm = fill_price * qty * self.commission_rate
            total_slippage = slippage_amt * qty
            proceeds = (fill_price * qty) - comm
            self.cash += proceeds

            gross_pnl = (fill_price - self.avg_entry_price) * qty
            net_pnl = gross_pnl - comm - total_slippage
            ret_pct = (fill_price - self.avg_entry_price) / (self.avg_entry_price + 1e-8)

            trade = TradeRecord(
                session_id=self.session_id,
                symbol=self.symbol,
                entry_time=self.entry_timestamp or order.timestamp,
                exit_time=order.timestamp,
                side="LONG",
                quantity=qty,
                entry_price=self.avg_entry_price,
                exit_price=fill_price,
                slippage=total_slippage,
                commission=comm,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                return_pct=ret_pct,
                exit_reason="Signal Exit",
            )
            self.trades.append(trade)

            self.position_qty -= qty
            if self.position_qty <= 0:
                self.position_qty = 0
                self.avg_entry_price = 0.0
                self.entry_timestamp = None
            net_cost = proceeds

        fill = FillEvent(
            timestamp=order.timestamp,
            symbol=self.symbol,
            direction=order.direction,
            quantity=qty,
            fill_price=fill_price,
            commission=comm,
            slippage=total_slippage,
            net_cost=net_cost,
        )
        self.fill_events.append(fill)
        return fill

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.reset()
        feat_df = compute_all_features(df)

        for idx in range(len(feat_df)):
            row = feat_df.iloc[idx]
            timestamp = row.name if hasattr(row, "name") else pd.Timestamp(idx)
            close = float(row["Close"])
            atr = float(row.get("atr_14", close * 0.01))

            cur_equity = self.cash + (self.position_qty * close)
            cb_tripped, _ = self.risk_manager.check_circuit_breaker(timestamp, cur_equity)

            sig = self.strategy.on_bar(row, symbol=self.symbol)

            if sig.signal_type == SignalType.BUY and self.position_qty == 0:
                if not cb_tripped:
                    # Estimate rolling Kelly edge if we have trade history
                    wr, wl = None, None
                    if len(self.trades) >= 5:
                        wins = [t.net_pnl for t in self.trades if t.net_pnl > 0]
                        losses = [abs(t.net_pnl) for t in self.trades if t.net_pnl < 0]
                        if wins and losses:
                            wr = len(wins) / len(self.trades)
                            wl = np.mean(wins) / (np.mean(losses) + 1e-8)

                    shares = self.risk_manager.calculate_position_size(
                        current_equity=cur_equity,
                        entry_price=close,
                        stop_loss_price=sig.stop_loss,
                        atr=atr,
                        win_rate=wr,
                        win_loss_ratio=wl,
                    )

                    # Limit by available cash
                    max_shares = int(self.cash / (close * (1 + self.slippage_rate + self.commission_rate)))
                    shares = min(shares, max_shares)

                    if shares > 0:
                        order = OrderEvent(
                            timestamp=timestamp,
                            symbol=self.symbol,
                            order_type="MARKET",
                            direction="BUY",
                            quantity=shares,
                            target_price=close,
                            stop_loss=sig.stop_loss,
                        )
                        self.execute_order(order)

            elif sig.signal_type == SignalType.SELL and self.position_qty > 0:
                order = OrderEvent(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    order_type="MARKET",
                    direction="SELL",
                    quantity=self.position_qty,
                    target_price=close,
                )
                self.execute_order(order)

            # Record end of bar equity
            bar_equity = self.cash + (self.position_qty * close)
            self.equity_curve.append(
                {
                    "timestamp": timestamp,
                    "cash": self.cash,
                    "position_qty": self.position_qty,
                    "close": close,
                    "equity": bar_equity,
                    "drawdown": 0.0,
                }
            )

        eq_df = pd.DataFrame(self.equity_curve)
        if not eq_df.empty:
            eq_df = eq_df.set_index("timestamp")
            eq_df["peak"] = eq_df["equity"].cummax()
            eq_df["drawdown"] = (eq_df["peak"] - eq_df["equity"]) / eq_df["peak"]

        metrics = self._calculate_metrics(eq_df, feat_df)

        if self.audit_to_db and self.trades:
            records = [
                {
                    "session_id": t.session_id,
                    "symbol": t.symbol,
                    "entry_time": str(t.entry_time),
                    "exit_time": str(t.exit_time),
                    "side": t.side,
                    "quantity": t.quantity,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "slippage": t.slippage,
                    "commission": t.commission,
                    "gross_pnl": t.gross_pnl,
                    "net_pnl": t.net_pnl,
                    "return_pct": t.return_pct,
                    "exit_reason": t.exit_reason,
                    "execution_type": "BACKTEST",
                }
                for t in self.trades
            ]
            log_trades_bulk(records, db_path=self.db_path)

        return {
            "session_id": self.session_id,
            "metrics": metrics,
            "equity_df": eq_df,
            "trades": self.trades,
            "features_df": feat_df,
        }

    def _calculate_metrics(self, eq_df: pd.DataFrame, feat_df: pd.DataFrame) -> Dict[str, Any]:
        if eq_df.empty or len(eq_df) < 2:
            return {}

        start_eq = self.initial_capital
        end_eq = float(eq_df["equity"].iloc[-1])
        tot_ret = (end_eq - start_eq) / start_eq

        eq_df["daily_return"] = eq_df["equity"].pct_change().fillna(0.0)
        daily_rets = eq_df["daily_return"].values

        n_days = len(eq_df)
        years = max(n_days / 252.0, 1.0 / 252.0)
        cagr = ((end_eq / start_eq) ** (1.0 / years)) - 1.0 if end_eq > 0 else -1.0

        r_mean = np.mean(daily_rets)
        r_std = np.std(daily_rets, ddof=1) if len(daily_rets) > 1 else 0.0
        sharpe = (np.sqrt(252) * (r_mean / r_std)) if r_std > 1e-8 else 0.0

        downside = daily_rets[daily_rets < 0]
        d_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
        sortino = (np.sqrt(252) * (r_mean / d_std)) if d_std > 1e-8 else 0.0

        max_dd = float(eq_df["drawdown"].max())

        bnh_start = float(feat_df["Close"].iloc[0])
        bnh_end = float(feat_df["Close"].iloc[-1])
        bnh_ret = (bnh_end - bnh_start) / bnh_start
        bnh_cagr = ((bnh_end / bnh_start) ** (1.0 / years)) - 1.0 if bnh_end > 0 else -1.0

        total_trades = len(self.trades)
        wins = [t for t in self.trades if t.net_pnl > 0]
        losses = [t for t in self.trades if t.net_pnl <= 0]
        win_rate = (len(wins) / total_trades) if total_trades > 0 else 0.0

        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-8 else (float("inf") if gross_profit > 0 else 0.0)
        avg_trade_pnl = (sum(t.net_pnl for t in self.trades) / total_trades) if total_trades > 0 else 0.0
        calmar = (cagr / max_dd) if max_dd > 1e-8 else 0.0

        return {
            "initial_capital": start_eq,
            "final_equity": end_eq,
            "total_return": tot_ret,
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "avg_trade_pnl": avg_trade_pnl,
            "benchmark_return": bnh_ret,
            "benchmark_cagr": bnh_cagr,
        }


def run_cli_backtest():
    parser = argparse.ArgumentParser(description="AlphaPulse Backtester")
    parser.add_argument("--symbol", type=str, default="SPY", help="Ticker symbol (e.g. SPY, AAPL)")
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial portfolio capital")
    parser.add_argument("--offline", action="store_true", help="Force synthetic data generator")
    args = parser.parse_args()

    print(f"\n--- AlphaPulse Backtester: {args.symbol} ({args.start} to {args.end}) ---")
    loader = DataLoader()
    df = loader.fetch_historical_data(symbol=args.symbol, start=args.start, end=args.end, offline=args.offline)

    bt = EventDrivenBacktester(symbol=args.symbol, initial_capital=args.capital)
    res = bt.run(df)
    m = res["metrics"]

    print(f"Final Equity:       ${m.get('final_equity', 0):,.2f}")
    print(f"Strategy Return:    {m.get('total_return', 0) * 100:.2f}% (CAGR: {m.get('cagr', 0) * 100:.2f}%)")
    print(f"Benchmark Return:   {m.get('benchmark_return', 0) * 100:.2f}% (CAGR: {m.get('benchmark_cagr', 0) * 100:.2f}%)")
    print(f"Sharpe Ratio:       {m.get('sharpe_ratio', 0):.2f}")
    print(f"Sortino Ratio:      {m.get('sortino_ratio', 0):.2f}")
    print(f"Max Drawdown:       {m.get('max_drawdown', 0) * 100:.2f}%")
    print(f"Total Trades:       {m.get('total_trades', 0)}")
    print(f"Win Rate:           {m.get('win_rate', 0) * 100:.2f}%")
    pf = f"{m.get('profit_factor', 0):.2f}" if m.get("profit_factor") != float("inf") else "inf"
    print(f"Profit Factor:      {pf}")
    print(f"Avg Trade P&L:      ${m.get('avg_trade_pnl', 0):,.2f}\n")


if __name__ == "__main__":
    run_cli_backtest()
