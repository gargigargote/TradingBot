"""Unit tests for Event-Driven Backtester and SQLite Trade Auditing."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile

from src.backtester import (
    EventDrivenBacktester,
    OrderEvent,
    EventType,
)
from src.data_loader import generate_synthetic_ohlcv
from src.audit_logger import init_audit_db, load_trades_from_db


def test_order_execution_slippage_and_fees():
    """Verify transaction costs and slippage (0.05%) applied to orders."""
    bt = EventDrivenBacktester(
        symbol="AAPL",
        initial_capital=100000.0,
        slippage_rate=0.0005,    # 0.05%
        commission_rate=0.0005,  # 0.05%
        audit_to_db=False,
    )
    
    # 1. Execute BUY order: 100 shares @ $100.0
    # Slippage: +0.05% -> Fill Price = 100 * 1.0005 = 100.05
    # Value = 10,005.0
    # Commission: 0.05% of 10,005 = 5.0025
    # Total Cash Outflow = 10,005 + 5.0025 = 10,010.0025
    buy_order = OrderEvent(
        timestamp=pd.Timestamp("2023-01-02"),
        symbol="AAPL",
        order_type="MARKET",
        direction="BUY",
        quantity=100,
        target_price=100.0,
    )
    buy_fill = bt.execute_order(buy_order)
    
    assert buy_fill.fill_price == pytest.approx(100.05)
    assert buy_fill.commission == pytest.approx(5.0025)
    assert bt.position_qty == 100
    assert bt.cash == pytest.approx(100000.0 - 10010.0025)
    
    # 2. Execute SELL order: 100 shares @ $110.0
    # Slippage: -0.05% -> Fill Price = 110 * 0.9995 = 109.945
    # Value = 10,994.5
    # Commission: 0.05% of 10,994.5 = 5.49725
    # Total Cash Inflow = 10,994.5 - 5.49725 = 10,989.00275
    sell_order = OrderEvent(
        timestamp=pd.Timestamp("2023-01-10"),
        symbol="AAPL",
        order_type="MARKET",
        direction="SELL",
        quantity=100,
        target_price=110.0,
    )
    sell_fill = bt.execute_order(sell_order)
    
    assert sell_fill.fill_price == pytest.approx(109.945)
    assert bt.position_qty == 0
    assert len(bt.trades) == 1
    
    trade = bt.trades[0]
    # Gross PnL = (109.945 - 100.05) * 100 = 989.5
    # Net PnL = 989.5 - (commission + slippage)
    assert trade.gross_pnl == pytest.approx(989.5)
    assert trade.net_pnl > 0


def test_full_backtest_simulation_and_metrics(tmp_path):
    """Verify end-to-end backtesting simulation on synthetic data and SQLite audit."""
    df = generate_synthetic_ohlcv(symbol="SPY", start="2022-01-01", end="2023-01-01", seed=123)
    test_db_path = tmp_path / "test_trades.db"
    
    init_audit_db(test_db_path)
    bt = EventDrivenBacktester(
        symbol="SPY",
        initial_capital=100000.0,
        audit_to_db=True,
        db_path=test_db_path,
    )
    results = bt.run(df)
    
    metrics = results["metrics"]
    assert "cagr" in metrics
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "win_rate" in metrics
    assert "profit_factor" in metrics
    
    equity_df = results["equity_df"]
    assert not equity_df.empty
    assert len(equity_df) == len(df)
    assert "equity" in equity_df.columns
    assert "drawdown" in equity_df.columns
    
    # Verify SQLite audit table has records
    audited_trades = load_trades_from_db(db_path=test_db_path)
    assert isinstance(audited_trades, pd.DataFrame)
    assert len(audited_trades) == len(results["trades"])
