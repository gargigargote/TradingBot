import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load environment configuration if present
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

# Alpaca paper credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Default trading & backtesting parameters
DEFAULT_INITIAL_CAPITAL = float(os.getenv("DEFAULT_INITIAL_CAPITAL", "100000.0"))
DEFAULT_RISK_PER_TRADE = float(os.getenv("DEFAULT_RISK_PER_TRADE", "0.02"))  # 2% equity at risk per trade
DEFAULT_KELLY_FRACTION = float(os.getenv("DEFAULT_KELLY_FRACTION", "0.5"))   # half-Kelly multiplier
MAX_DAILY_DRAWDOWN_LIMIT = float(os.getenv("MAX_DAILY_DRAWDOWN_LIMIT", "0.03"))  # 3% circuit breaker
DEFAULT_SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0005"))         # 5 bps
DEFAULT_COMMISSION_RATE = float(os.getenv("COMMISSION_RATE", "0.0005"))     # 5 bps

# Storage & runtime options
SQLITE_DB_PATH = Path(os.getenv("SQLITE_DB_PATH", str(DATA_DIR / "trades_audit.db")))
USE_OFFLINE_DATA_FALLBACK = os.getenv("USE_OFFLINE_DATA_FALLBACK", "True").lower() in ("true", "1", "yes")

LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)


def setup_logger(name: str = "alphapulse") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(LOG_LEVEL)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
        log.addHandler(handler)
    return log


logger = setup_logger()
