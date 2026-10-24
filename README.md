# AlphaPulse

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://alphapulse-trading-bot.streamlit.app/)
[![CI](https://github.com/gargigargote/TradingBot/actions/workflows/tests.yml/badge.svg)](https://github.com/gargigargote/TradingBot/actions/workflows/tests.yml)

🌐 **Live Terminal:** [https://alphapulse-trading-bot.streamlit.app/](https://alphapulse-trading-bot.streamlit.app/)

A modular, event-driven algorithmic trading, backtesting, and paper-trading system written in Python.

AlphaPulse implements a dual-layer strategy:
1. **Regime Filter:** Uses 50/200 EMA crossover and normalized ATR to detect trend vs. mean-reverting / high-volatility chop regimes.
2. **Mean-Reversion Execution:** Exploits Bollinger Band excursions using 20-day rolling Z-scores ($Z \le -2.0$) with dynamic ATR trailing stops.

Includes dynamic position sizing (fixed fractional + half-Kelly), a 3% intraday circuit breaker, fee & slippage modeling (5 bps default), trade audit logging to SQLite, and an interactive Streamlit analytics dashboard.

---

## Architecture Overview

```
alphapulse/
├── .github/workflows/
│   └── tests.yml             # CI workflow (pytest on Python 3.11/3.12)
├── data/
│   └── trades_audit.db       # SQLite trade audit database
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── src/
│   ├── config.py             # Settings, env vars, and logging
│   ├── data_loader.py        # Yahoo Finance ingestion + synthetic fallback
│   ├── features.py           # Technical indicators (ATR, EMA, SMA, Z-score, Bollinger)
│   ├── strategy.py           # Regime classifier & Bollinger Z-score logic
│   ├── risk_manager.py       # Position sizing, Kelly multiplier & 3% daily circuit breaker
│   ├── audit_logger.py       # SQLite trade audit logger
│   ├── execution.py          # Alpaca paper trading & mock executor
│   └── backtester.py         # Event-driven backtesting engine & CLI runner
├── tests/
│   ├── test_features.py
│   ├── test_strategy.py
│   ├── test_risk_manager.py
│   └── test_backtester.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Strategy & Risk Mechanics

### 1. Market Regime Classification
- **Bull Trend**: $\text{EMA}_{50} \ge \text{EMA}_{200}$ and ATR expansion ratio $\le 1.25$.
- **Bear Trend**: $\text{EMA}_{50} < \text{EMA}_{200}$.
- **High Volatility Chop**: $\text{ATR}_{14} / \text{ATR}_{50} > 1.25$ (halts new entries).
- **Low Volatility Mean Reversion**: Sideways market with contained volatility.

### 2. Entry & Exit Logic
- **Entry**: Buy when $Z \le -2.0$ (oversold excursion relative to 20-day MA) in favorable regime.
- **Exit**: Sell when price reverts to mean ($Z \ge 0.0$) or when trailing ATR stop ($2.0 \times \text{ATR}$) is breached.

### 3. Sizing & Circuit Breaker
- **Position Sizing**:
  $$\text{Shares} = \left\lfloor \frac{\text{Capital} \times \text{Risk\%}}{\max(|P_{\text{entry}} - P_{\text{stop}}|, 0.5 \times \text{ATR})} \right\rfloor$$
- **Half-Kelly Scaling**: Bounded by $0.5 \cdot f^*$, where $f^* = p - \frac{1-p}{b}$.
- **3% Circuit Breaker**: If intraday portfolio drawdown from the day's open or peak reaches 3%, trading pauses for the rest of the day.

---

## Setup & Quickstart

### 1. Installation

```bash
# Clone and create venv
cd alphapulse
python -m venv venv

# Activate venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests

```bash
pytest tests/ -v
```

### 3. Run a Backtest via CLI

```bash
# Backtest on SPY using Yahoo Finance data
python -m src.backtester --symbol SPY --start 2022-01-01 --end 2024-01-01

# Run offline using synthetic geometric random walk
python -m src.backtester --symbol AAPL --offline
```

### 4. Launch the Dashboard

You can access the live deployed dashboard directly at:
👉 **[https://alphapulse-trading-bot.streamlit.app/](https://alphapulse-trading-bot.streamlit.app/)**

Or run it locally:
```bash
streamlit run dashboard/app.py
```
Access locally at `http://localhost:8501`. Features include:
- Interactive Plotly equity curves and underwater drawdowns.
- Price charts with Bollinger Bands, 50/200 EMAs, and buy/sell execution markers.
- Completed trade history with 1-click CSV download.
- Alpaca paper trading account balance, open positions, and order placement.
- SQLite trade audit log browser.
