from typing import Any, Dict, List, Optional
import pandas as pd
import requests

from src.audit_logger import log_trade_to_db
from src.config import ALPACA_API_KEY, ALPACA_BASE_URL, ALPACA_SECRET_KEY, logger


class AlpacaExecutionHandler:
    """Connects to Alpaca paper trading API, with in-memory mock fallback when keys aren't set."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        mock_mode: bool = False,
    ):
        self.api_key = api_key or ALPACA_API_KEY
        self.secret_key = secret_key or ALPACA_SECRET_KEY
        self.base_url = (base_url or ALPACA_BASE_URL).rstrip("/")

        # Automatically use mock if credentials look like placeholders
        self.mock_mode = mock_mode or (not self.api_key or "your_" in self.api_key.lower())

        # Internal state for mock mode
        self._mock_cash = 100000.0
        self._mock_portfolio_val = 100000.0
        self._mock_positions: Dict[str, Dict[str, Any]] = {}
        self._mock_orders: List[Dict[str, Any]] = []

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    def get_account(self) -> Dict[str, Any]:
        if self.mock_mode:
            return {
                "id": "mock-paper-account",
                "status": "ACTIVE (MOCK)",
                "currency": "USD",
                "cash": f"{self._mock_cash:.2f}",
                "portfolio_value": f"{self._mock_portfolio_val:.2f}",
                "buying_power": f"{self._mock_cash * 2:.2f}",
                "equity": f"{self._mock_portfolio_val:.2f}",
            }

        url = f"{self.base_url}/v2/account"
        try:
            r = requests.get(url, headers=self._headers(), timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as err:
            logger.warning(f"Error fetching Alpaca account: {err}")
            return {
                "status": "DISCONNECTED",
                "cash": "0.00",
                "portfolio_value": "0.00",
                "buying_power": "0.00",
                "error": str(err),
            }

    def get_positions(self) -> List[Dict[str, Any]]:
        if self.mock_mode:
            return list(self._mock_positions.values())

        url = f"{self.base_url}/v2/positions"
        try:
            r = requests.get(url, headers=self._headers(), timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as err:
            logger.warning(f"Error fetching Alpaca positions: {err}")
            return []

    def get_orders(self, status: str = "all", limit: int = 50) -> List[Dict[str, Any]]:
        if self.mock_mode:
            return self._mock_orders[-limit:]

        url = f"{self.base_url}/v2/orders"
        try:
            r = requests.get(url, headers=self._headers(), params={"status": status, "limit": limit}, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as err:
            logger.warning(f"Error fetching Alpaca orders: {err}")
            return []

    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str = "buy",
        order_type: str = "market",
        time_in_force: str = "day",
        current_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        side = side.lower()
        symbol = symbol.upper()

        if self.mock_mode:
            price = current_price or 150.0
            cost = price * qty
            order_id = f"mock-ord-{len(self._mock_orders) + 1:04d}"
            now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

            order_data = {
                "id": order_id,
                "symbol": symbol,
                "qty": str(qty),
                "side": side,
                "type": order_type,
                "status": "filled",
                "filled_avg_price": f"{price:.2f}",
                "filled_at": now_str,
            }
            self._mock_orders.append(order_data)

            if side == "buy":
                self._mock_cash -= cost
                if symbol in self._mock_positions:
                    pos = self._mock_positions[symbol]
                    old_qty = int(pos["qty"])
                    old_cost = float(pos["avg_entry_price"]) * old_qty
                    new_qty = old_qty + qty
                    pos["qty"] = str(new_qty)
                    pos["avg_entry_price"] = f"{(old_cost + cost) / new_qty:.2f}"
                    pos["market_value"] = f"{new_qty * price:.2f}"
                else:
                    self._mock_positions[symbol] = {
                        "symbol": symbol,
                        "qty": str(qty),
                        "avg_entry_price": f"{price:.2f}",
                        "current_price": f"{price:.2f}",
                        "market_value": f"{cost:.2f}",
                        "unrealized_pl": "0.00",
                    }
            elif side == "sell":
                self._mock_cash += cost
                if symbol in self._mock_positions:
                    pos = self._mock_positions[symbol]
                    old_qty = int(pos["qty"])
                    entry_p = float(pos["avg_entry_price"])
                    closed_qty = min(qty, old_qty)
                    pnl = (price - entry_p) * closed_qty
                    rem = max(0, old_qty - qty)

                    log_trade_to_db(
                        {
                            "session_id": "alpaca_mock",
                            "symbol": symbol,
                            "entry_time": now_str,
                            "exit_time": now_str,
                            "side": "LONG_EXIT",
                            "quantity": closed_qty,
                            "entry_price": entry_p,
                            "exit_price": price,
                            "slippage": 0.0,
                            "commission": 0.0,
                            "gross_pnl": pnl,
                            "net_pnl": pnl,
                            "return_pct": (price - entry_p) / entry_p,
                            "exit_reason": "Paper market sell",
                            "execution_type": "PAPER",
                        }
                    )

                    if rem == 0:
                        del self._mock_positions[symbol]
                    else:
                        pos["qty"] = str(rem)
                        pos["market_value"] = f"{rem * price:.2f}"

            pos_val = sum(float(p["market_value"]) for p in self._mock_positions.values())
            self._mock_portfolio_val = self._mock_cash + pos_val
            return order_data

        url = f"{self.base_url}/v2/orders"
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        try:
            r = requests.post(url, headers=self._headers(), json=payload, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as err:
            logger.error(f"Error submitting order: {err}")
            return {"status": "error", "message": str(err)}

    def cancel_all_orders(self) -> bool:
        if self.mock_mode:
            self._mock_orders = [o for o in self._mock_orders if o.get("status") == "filled"]
            return True

        url = f"{self.base_url}/v2/orders"
        try:
            r = requests.delete(url, headers=self._headers(), timeout=10)
            r.raise_for_status()
            return True
        except Exception as err:
            logger.error(f"Error canceling orders: {err}")
            return False
