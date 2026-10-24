import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from src.config import SQLITE_DB_PATH, logger


def init_audit_db(db_path: Path = SQLITE_DB_PATH) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                slippage REAL NOT NULL,
                commission REAL NOT NULL,
                gross_pnl REAL NOT NULL,
                net_pnl REAL NOT NULL,
                return_pct REAL NOT NULL,
                exit_reason TEXT,
                execution_type TEXT DEFAULT 'BACKTEST',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    return db_path


def log_trade_to_db(trade: Dict[str, Any], db_path: Path = SQLITE_DB_PATH):
    init_audit_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO trades (
                session_id, symbol, entry_time, exit_time, side, quantity,
                entry_price, exit_price, slippage, commission, gross_pnl,
                net_pnl, return_pct, exit_reason, execution_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(trade.get("session_id", "default")),
                str(trade.get("symbol", "ASSET")),
                str(trade.get("entry_time", "")),
                str(trade.get("exit_time", "")),
                str(trade.get("side", "LONG")),
                int(trade.get("quantity", 0)),
                float(trade.get("entry_price", 0.0)),
                float(trade.get("exit_price", 0.0)),
                float(trade.get("slippage", 0.0)),
                float(trade.get("commission", 0.0)),
                float(trade.get("gross_pnl", 0.0)),
                float(trade.get("net_pnl", 0.0)),
                float(trade.get("return_pct", 0.0)),
                str(trade.get("exit_reason", "")),
                str(trade.get("execution_type", "BACKTEST")),
            ),
        )
        conn.commit()


def log_trades_bulk(trades: List[Dict[str, Any]], db_path: Path = SQLITE_DB_PATH):
    if not trades:
        return
    init_audit_db(db_path)
    rows = [
        (
            str(t.get("session_id", "default")),
            str(t.get("symbol", "ASSET")),
            str(t.get("entry_time", "")),
            str(t.get("exit_time", "")),
            str(t.get("side", "LONG")),
            int(t.get("quantity", 0)),
            float(t.get("entry_price", 0.0)),
            float(t.get("exit_price", 0.0)),
            float(t.get("slippage", 0.0)),
            float(t.get("commission", 0.0)),
            float(t.get("gross_pnl", 0.0)),
            float(t.get("net_pnl", 0.0)),
            float(t.get("return_pct", 0.0)),
            str(t.get("exit_reason", "")),
            str(t.get("execution_type", "BACKTEST")),
        )
        for t in trades
    ]
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO trades (
                session_id, symbol, entry_time, exit_time, side, quantity,
                entry_price, exit_price, slippage, commission, gross_pnl,
                net_pnl, return_pct, exit_reason, execution_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def load_trades_from_db(
    session_id: Optional[str] = None,
    db_path: Path = SQLITE_DB_PATH,
) -> pd.DataFrame:
    init_audit_db(db_path)
    with sqlite3.connect(db_path) as conn:
        if session_id:
            query = "SELECT * FROM trades WHERE session_id = ? ORDER BY id DESC"
            return pd.read_sql_query(query, conn, params=(session_id,))
        query = "SELECT * FROM trades ORDER BY id DESC"
        return pd.read_sql_query(query, conn)
