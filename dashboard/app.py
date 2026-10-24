import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.audit_logger import load_trades_from_db
from src.backtester import EventDrivenBacktester
from src.config import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_KELLY_FRACTION,
    DEFAULT_RISK_PER_TRADE,
    SQLITE_DB_PATH,
)
from src.data_loader import DataLoader
from src.execution import AlpacaExecutionHandler
from src.risk_manager import RiskManager
from src.strategy import DualLayerRegimeBollingerStrategy

st.set_page_config(
    page_title="AlphaPulse | Quantitative Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom minimal dark styling
st.markdown(
    """
    <style>
    .terminal-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 2px;
    }
    .terminal-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 20px;
    }
    .stat-box {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 12px 14px;
        text-align: center;
    }
    .stat-val {
        font-size: 1.4rem;
        font-weight: 600;
        color: #F1F5F9;
    }
    .stat-lbl {
        font-size: 0.78rem;
        color: #64748B;
        text-transform: uppercase;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="terminal-title">AlphaPulse Quantitative Terminal</div>', unsafe_allow_html=True)
st.markdown('<div class="terminal-subtitle">Dual-layer market regime filter & Bollinger Z-score mean-reversion engine</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### Settings")
mode = st.sidebar.radio("View", ["Backtester", "Alpaca Paper Trading", "Trade Audit Log"])

st.sidebar.markdown("---")
st.sidebar.markdown("#### Market Data")
ticker = st.sidebar.selectbox("Ticker", ["SPY", "AAPL", "QQQ", "NVDA", "MSFT", "TSLA", "Custom"])
if ticker == "Custom":
    symbol = st.sidebar.text_input("Symbol", value="SPY").upper()
else:
    symbol = ticker

col_start, col_end = st.sidebar.columns(2)
start_date = col_start.date_input("Start", value=pd.to_datetime("2022-01-01"))
end_date = col_end.date_input("End", value=pd.to_datetime("2024-01-01"))
use_synthetic = st.sidebar.checkbox("Use synthetic data (offline)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Strategy & Sizing")
capital = st.sidebar.number_input("Starting Capital ($)", min_value=1000.0, max_value=10_000_000.0, value=DEFAULT_INITIAL_CAPITAL, step=5000.0)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, DEFAULT_RISK_PER_TRADE * 100, 0.5) / 100.0
kelly_mult = st.sidebar.slider("Kelly Fraction", 0.1, 1.0, DEFAULT_KELLY_FRACTION, 0.1)
z_thresh = st.sidebar.slider("Entry Z-Score", -3.5, -1.0, -2.0, 0.1)
atr_stop = st.sidebar.slider("ATR Trailing Stop Mult", 1.0, 4.0, 2.0, 0.5)
allow_bear = st.sidebar.checkbox("Allow Bear Regime Entries", value=False)


# -------------------------------------------------------------------------
# VIEW 1: BACKTESTER
# -------------------------------------------------------------------------
if mode == "Backtester":
    if st.sidebar.button("Run Backtest", type="primary", use_container_width=True) or "bt_data" not in st.session_state:
        with st.spinner("Running simulation..."):
            loader = DataLoader()
            df = loader.fetch_historical_data(
                symbol=symbol,
                start=str(start_date),
                end=str(end_date),
                offline=use_synthetic,
            )

            strat = DualLayerRegimeBollingerStrategy(
                z_entry_threshold=z_thresh,
                z_exit_threshold=0.0,
                atr_stop_multiplier=atr_stop,
                allow_bear_entries=allow_bear,
            )
            rm = RiskManager(
                initial_capital=capital,
                risk_per_trade=risk_pct,
                fractional_kelly=kelly_mult,
            )
            bt = EventDrivenBacktester(
                symbol=symbol,
                initial_capital=capital,
                strategy=strat,
                risk_manager=rm,
                audit_to_db=True,
            )
            st.session_state["bt_data"] = bt.run(df)
            st.session_state["active_sym"] = symbol

    res = st.session_state.get("bt_data")
    if res and "metrics" in res:
        m = res["metrics"]
        eq_df = res["equity_df"]
        feat_df = res["features_df"]
        trades = res["trades"]

        # KPI row
        k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
        ret_val = m.get("total_return", 0) * 100
        ret_color = "#10B981" if ret_val >= 0 else "#EF4444"

        with k1:
            st.markdown(f'<div class="stat-box"><div class="stat-val" style="color:{ret_color}">{ret_val:.1f}%</div><div class="stat-lbl">Return</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{m.get("cagr", 0)*100:.1f}%</div><div class="stat-lbl">CAGR</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{m.get("sharpe_ratio", 0):.2f}</div><div class="stat-lbl">Sharpe</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{m.get("sortino_ratio", 0):.2f}</div><div class="stat-lbl">Sortino</div></div>', unsafe_allow_html=True)
        with k5:
            st.markdown(f'<div class="stat-box"><div class="stat-val" style="color:#EF4444">-{m.get("max_drawdown", 0)*100:.1f}%</div><div class="stat-lbl">Max DD</div></div>', unsafe_allow_html=True)
        with k6:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{m.get("win_rate", 0)*100:.1f}%</div><div class="stat-lbl">Win Rate</div></div>', unsafe_allow_html=True)
        with k7:
            pf = f"{m.get('profit_factor', 0):.2f}" if m.get("profit_factor") != float("inf") else "inf"
            st.markdown(f'<div class="stat-box"><div class="stat-val">{pf}</div><div class="stat-lbl">Profit Factor</div></div>', unsafe_allow_html=True)
        with k8:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{m.get("total_trades", 0)}</div><div class="stat-lbl">Trades</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Equity Curve and Underwater profile
        bnh_shares = capital / float(feat_df["Close"].iloc[0])
        bnh_curve = bnh_shares * feat_df["Close"]

        fig_eq = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.72, 0.28],
            subplot_titles=["Portfolio Value ($) vs Buy & Hold Benchmark", "Drawdown Profile (%)"],
        )
        fig_eq.add_trace(
            go.Scatter(x=eq_df.index, y=eq_df["equity"], name="AlphaPulse", line=dict(color="#38BDF8", width=2.2)),
            row=1, col=1,
        )
        fig_eq.add_trace(
            go.Scatter(x=feat_df.index, y=bnh_curve, name=f"{symbol} Benchmark", line=dict(color="#64748B", width=1.5, dash="dash")),
            row=1, col=1,
        )
        fig_eq.add_trace(
            go.Scatter(x=eq_df.index, y=-eq_df["drawdown"] * 100, name="Drawdown %", fill="tozeroy", line=dict(color="#EF4444", width=1.2), fillcolor="rgba(239, 68, 68, 0.15)"),
            row=2, col=1,
        )
        fig_eq.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=30, t=40, b=30), hovermode="x unified")
        st.plotly_chart(fig_eq, use_container_width=True)

        # Price, Bollinger Bands, and Execution Signals
        fig_bars = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.72, 0.28],
            subplot_titles=[f"{symbol} Price, Bollinger Bands & Executions", "Z-Score (20-day MA)"],
        )
        fig_bars.add_trace(go.Scatter(x=feat_df.index, y=feat_df["Close"], name="Close", line=dict(color="#E2E8F0", width=1.4)), row=1, col=1)
        fig_bars.add_trace(go.Scatter(x=feat_df.index, y=feat_df["bb_upper"], name="Upper BB", line=dict(color="rgba(56, 189, 248, 0.4)", width=1, dash="dot")), row=1, col=1)
        fig_bars.add_trace(go.Scatter(x=feat_df.index, y=feat_df["bb_lower"], name="Lower BB", line=dict(color="rgba(56, 189, 248, 0.4)", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(56, 189, 248, 0.05)"), row=1, col=1)
        fig_bars.add_trace(go.Scatter(x=feat_df.index, y=feat_df["ema_50"], name="50 EMA", line=dict(color="#F59E0B", width=1.2)), row=1, col=1)
        fig_bars.add_trace(go.Scatter(x=feat_df.index, y=feat_df["ema_200"], name="200 EMA", line=dict(color="#8B5CF6", width=1.2)), row=1, col=1)

        # Trade markers
        if trades:
            fig_bars.add_trace(
                go.Scatter(
                    x=[t.entry_time for t in trades],
                    y=[t.entry_price for t in trades],
                    mode="markers",
                    name="Buy",
                    marker=dict(symbol="triangle-up", size=10, color="#10B981"),
                ),
                row=1, col=1,
            )
            fig_bars.add_trace(
                go.Scatter(
                    x=[t.exit_time for t in trades],
                    y=[t.exit_price for t in trades],
                    mode="markers",
                    name="Sell",
                    marker=dict(symbol="triangle-down", size=10, color="#EF4444"),
                ),
                row=1, col=1,
            )

        # Z-Score indicator trace
        fig_bars.add_trace(go.Scatter(x=feat_df.index, y=feat_df["z_score_20"], name="Z-Score", line=dict(color="#38BDF8", width=1.3)), row=2, col=1)
        fig_bars.add_hline(y=z_thresh, line_dash="dash", line_color="#10B981", row=2, col=1)
        fig_bars.add_hline(y=0.0, line_dash="dot", line_color="#64748B", row=2, col=1)
        fig_bars.update_layout(template="plotly_dark", height=540, margin=dict(l=40, r=30, t=40, b=30), hovermode="x unified")
        st.plotly_chart(fig_bars, use_container_width=True)

        # Trades table
        st.markdown("#### Completed Trades")
        if trades:
            trade_rows = [
                {
                    "Symbol": t.symbol,
                    "Entry Time": str(t.entry_time)[:19],
                    "Exit Time": str(t.exit_time)[:19],
                    "Qty": t.quantity,
                    "Entry ($)": round(t.entry_price, 2),
                    "Exit ($)": round(t.exit_price, 2),
                    "Slippage ($)": round(t.slippage, 2),
                    "Commission ($)": round(t.commission, 2),
                    "Net PnL ($)": round(t.net_pnl, 2),
                    "Return (%)": round(t.return_pct * 100, 2),
                    "Exit Reason": t.exit_reason,
                }
                for t in trades
            ]
            t_df = pd.DataFrame(trade_rows)
            st.dataframe(t_df, use_container_width=True, height=260)

            # 1-click CSV download
            csv_bytes = t_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Trades CSV",
                data=csv_bytes,
                file_name=f"{symbol}_trades.csv",
                mime="text/csv",
            )
        else:
            st.info("No trades executed with current parameters.")


# -------------------------------------------------------------------------
# VIEW 2: ALPACA PAPER TRADING
# -------------------------------------------------------------------------
elif mode == "Alpaca Paper Trading":
    st.markdown("### Alpaca Paper Trading")
    alpaca = AlpacaExecutionHandler()
    acc = alpaca.get_account()

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Status", acc.get("status", "OFFLINE"))
    a2.metric("Portfolio Value", f"${float(acc.get('portfolio_value', 0)):,.2f}")
    a3.metric("Cash", f"${float(acc.get('cash', 0)):,.2f}")
    a4.metric("Buying Power", f"${float(acc.get('buying_power', 0)):,.2f}")

    st.markdown("---")
    c_order, c_pos = st.columns([1, 1])

    with c_order:
        st.markdown("#### Place Order")
        ord_sym = st.text_input("Symbol", value="SPY").upper()
        ord_qty = st.number_input("Shares", min_value=1, max_value=10000, value=10)
        ord_side = st.selectbox("Side", ["BUY", "SELL"])
        ord_px = st.number_input("Price ($)", min_value=1.0, value=500.0, step=1.0)

        if st.button("Submit Order", type="primary", use_container_width=True):
            res = alpaca.submit_order(
                symbol=ord_sym,
                qty=int(ord_qty),
                side=ord_side.lower(),
                current_price=ord_px,
            )
            st.success(f"Executed: {ord_side} {ord_qty} {ord_sym} @ ${ord_px:.2f}")
            st.rerun()

    with c_pos:
        st.markdown("#### Positions")
        positions = alpaca.get_positions()
        if positions:
            st.dataframe(pd.DataFrame(positions), use_container_width=True)
        else:
            st.info("No open positions.")

    st.markdown("---")
    st.markdown("#### Recent Orders")
    orders = alpaca.get_orders()
    if orders:
        st.dataframe(pd.DataFrame(orders), use_container_width=True)
    else:
        st.info("No orders yet.")


# -------------------------------------------------------------------------
# VIEW 3: TRADE AUDIT LOG
# -------------------------------------------------------------------------
elif mode == "Trade Audit Log":
    st.markdown("### SQLite Audit Database")
    st.caption(f"Path: `{SQLITE_DB_PATH}`")

    db_trades = load_trades_from_db()
    if not db_trades.empty:
        total_pnl = db_trades["net_pnl"].sum()
        win_rate = (db_trades["net_pnl"] > 0).mean() * 100
        commissions = db_trades["commission"].sum()
        slippage = db_trades["slippage"].sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cumulative Net P&L", f"${total_pnl:,.2f}")
        m2.metric("Win Rate", f"{win_rate:.1f}%")
        m3.metric("Total Commissions", f"${commissions:,.2f}")
        m4.metric("Total Slippage", f"${slippage:,.2f}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(db_trades, use_container_width=True, height=320)

        csv_data = db_trades.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Full Audit Log (CSV)",
            data=csv_data,
            file_name="trades_audit_full.csv",
            mime="text/csv",
        )
    else:
        st.info("No trades in database yet. Run a backtest or place a paper trade to log records.")
