"""
MoneyClaw Risk Manager — IMMUTABLE last line of defense.

This module enforces hard-coded risk rules that CANNOT be modified by the
evolution system or any runtime process.  Even if every strategy simultaneously
produces garbage signals, the RiskManager prevents catastrophic capital loss.

Rules are defined as class-level constants.  Any attempt to relax them must
go through a code review and deployment — never through an automated path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from moneyclaw.models import TradeSignal, Position, TradingState, OrderSide

logger = logging.getLogger(__name__)


class RiskManager:
    """Immutable risk gatekeeper for every trade the engine considers.

    IMMUTABLE_RULES are class-level constants.  They are not stored in a
    config file, database, or environment variable.  Changing them requires
    a source-code change, code review, and redeployment.
    """

    # ------------------------------------------------------------------
    # IMMUTABLE RULES — class constants, NOT runtime-configurable
    # ------------------------------------------------------------------
    IMMUTABLE_RULES: dict = {
        "max_risk_per_trade_pct": 0.02,      # Never risk > 2% per trade
        "max_position_pct": 0.12,            # Never > 12% in one coin
        "max_positions": 8,                   # Max 8 open positions
        "daily_loss_limit_pct": 0.10,        # Circuit breaker at 10% daily loss
        "max_drawdown_pct": 0.25,            # Hard stop at 25% from peak
        "min_reserve_pct": 0.20,             # Always keep 20% cash
        "min_reward_risk_ratio": 1.5,        # Minimum 1.5 R:R
    }

    def __init__(self, starting_capital: float = 100.0) -> None:
        self.starting_capital = starting_capital
        self.peak_value = starting_capital
        self.daily_pnl = 0.0
        self.state = TradingState.ACTIVE

    # ------------------------------------------------------------------
    # Core gate — every signal MUST pass through here
    # ------------------------------------------------------------------

    def validate_signal(
        self,
        signal: TradeSignal,
        portfolio_value: float,
        cash_balance: float,
        open_positions: list[Position],
    ) -> tuple[bool, str]:
        """Validate a trade signal against every immutable rule.

        Returns (True, "Approved") if the signal passes all checks, or
        (False, reason) on the FIRST failure encountered.
        """
        rules = self.IMMUTABLE_RULES

        # 1. Trading state must be ACTIVE
        if self.state != TradingState.ACTIVE:
            reason = f"Trading paused: {self.state.value}"
            logger.warning("Signal REJECTED [%s]: %s", signal.product_id, reason)
            return False, reason

        # 2. Max open positions
        if len(open_positions) >= rules["max_positions"]:
            reason = (
                f"Max positions reached: {len(open_positions)}/{rules['max_positions']}"
            )
            logger.warning("Signal REJECTED [%s]: %s", signal.product_id, reason)
            return False, reason

        # 3. Daily loss limit
        daily_loss_threshold = -(self.starting_capital * rules["daily_loss_limit_pct"])
        if self.daily_pnl <= daily_loss_threshold:
            reason = (
                f"Daily loss limit hit: ${self.daily_pnl:.2f} "
                f"<= ${daily_loss_threshold:.2f}"
            )
            logger.warning("Signal REJECTED [%s]: %s", signal.product_id, reason)
            return False, reason

        # 4. Max drawdown from peak
        drawdown = self.check_drawdown(portfolio_value)
        if drawdown >= rules["max_drawdown_pct"]:
            reason = (
                f"Max drawdown exceeded: {drawdown:.1%} >= "
                f"{rules['max_drawdown_pct']:.0%}"
            )
            logger.warning("Signal REJECTED [%s]: %s", signal.product_id, reason)
            return False, reason

        # 5. Stop loss MUST be set (non-zero)
        if signal.stop_loss <= 0:
            reason = "Signal has no stop_loss — REFUSED (all trades require a stop)"
            logger.warning("Signal REJECTED [%s]: %s", signal.product_id, reason)
            return False, reason

        # 6. Take profit MUST be set (non-zero)
        if signal.take_profit <= 0:
            reason = "Signal has no take_profit — REFUSED (all trades require a target)"
            logger.warning("Signal REJECTED [%s]: %s", signal.product_id, reason)
            return False, reason

        # 7. Reward-to-risk ratio
        risk_distance = abs(signal.entry_price - signal.stop_loss)
        reward_distance = abs(signal.take_profit - signal.entry_price)
        if risk_distance == 0:
            reason = "Stop loss equals entry price — zero risk distance"
            logger.warning("Signal REJECTED [%s]: %s", signal.product_id, reason)
            return False, reason

        rr_ratio = reward_distance / risk_distance
        if rr_ratio < rules["min_reward_risk_ratio"]:
            reason = (
                f"R:R ratio too low: {rr_ratio:.2f} < "
                f"{rules['min_reward_risk_ratio']}"
            )
            logger.warning("Signal REJECTED [%s]: %s", signal.product_id, reason)
            return False, reason

        # 8. Position size cap — estimated position value vs portfolio
        position_value = self.calculate_position_size(signal, portfolio_value, cash_balance)
        max_position_value = portfolio_value * rules["max_position_pct"]
        if position_value > max_position_value:
            reason = (
                f"Position would exceed max allocation: "
                f"${position_value:.2f} > ${max_position_value:.2f} "
                f"({rules['max_position_pct']:.0%} of portfolio)"
            )
            logger.warning("Signal REJECTED [%s]: %s", signal.product_id, reason)
            return False, reason

        # 9. Enough cash after maintaining reserve
        available = self.available_cash_for_trading(cash_balance, portfolio_value)
        if available <= 0:
            reason = (
                f"Insufficient cash after reserve: "
                f"${cash_balance:.2f} cash, "
                f"${portfolio_value * rules['min_reserve_pct']:.2f} reserve required"
            )
            logger.warning("Signal REJECTED [%s]: %s", signal.product_id, reason)
            return False, reason

        logger.info(
            "Signal APPROVED [%s %s]: R:R=%.2f, size=$%.2f",
            signal.side.value,
            signal.product_id,
            rr_ratio,
            position_value,
        )
        return True, "Approved"

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        signal: TradeSignal,
        portfolio_value: float,
        cash_balance: float,
    ) -> float:
        """Calculate the maximum position size in USD quote terms.

        Uses the fixed-fractional method: risk at most max_risk_per_trade_pct
        of portfolio value, then cap by max_position_pct and available cash.

        Returns 0.0 if any constraint makes the trade impossible.
        """
        rules = self.IMMUTABLE_RULES

        # Max dollar risk allowed
        max_risk_amount = portfolio_value * rules["max_risk_per_trade_pct"]

        # Stop distance as a fraction of entry price
        stop_distance = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
        if stop_distance == 0:
            return 0.0

        # Position size derived from risk budget
        size_from_risk = max_risk_amount / stop_distance

        # Cap at max allocation per coin
        size_cap_position = portfolio_value * rules["max_position_pct"]

        # Cap at available cash (after reserve)
        available = self.available_cash_for_trading(cash_balance, portfolio_value)

        # Take the most restrictive cap
        position_size = min(size_from_risk, size_cap_position, available)

        if position_size <= 0:
            return 0.0

        return position_size

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def check_circuit_breaker(
        self,
        daily_pnl: float,
        portfolio_value: float,
    ) -> TradingState:
        """Evaluate whether circuit breaker conditions are met.

        Updates internal state and returns the current TradingState.
        """
        self.daily_pnl = daily_pnl
        rules = self.IMMUTABLE_RULES

        # Daily loss limit
        daily_loss_threshold = -(self.starting_capital * rules["daily_loss_limit_pct"])
        if daily_pnl <= daily_loss_threshold:
            self.state = TradingState.CIRCUIT_BREAKER
            logger.critical(
                "CIRCUIT BREAKER TRIPPED — daily loss: $%.2f (<= $%.2f)",
                daily_pnl,
                daily_loss_threshold,
            )
            return self.state

        # Max drawdown from peak
        drawdown = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0.0
        if drawdown >= rules["max_drawdown_pct"]:
            self.state = TradingState.CIRCUIT_BREAKER
            logger.critical(
                "CIRCUIT BREAKER TRIPPED — drawdown: %.1f%% (peak $%.2f, now $%.2f)",
                drawdown * 100,
                self.peak_value,
                portfolio_value,
            )
            return self.state

        # Update peak if we have a new high
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        return self.state

    # ------------------------------------------------------------------
    # Drawdown calculation
    # ------------------------------------------------------------------

    def check_drawdown(self, portfolio_value: float) -> float:
        """Return current drawdown from peak as a decimal.

        0.0 = no drawdown (at or above peak).
        0.25 = 25% below peak.
        """
        if self.peak_value <= 0:
            return 0.0
        drawdown = (self.peak_value - portfolio_value) / self.peak_value
        return max(0.0, drawdown)

    # ------------------------------------------------------------------
    # Cash reserve
    # ------------------------------------------------------------------

    def available_cash_for_trading(
        self,
        cash_balance: float,
        portfolio_value: float,
    ) -> float:
        """Return cash available after subtracting the mandatory reserve."""
        reserve = portfolio_value * self.IMMUTABLE_RULES["min_reserve_pct"]
        return max(0.0, cash_balance - reserve)

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def reset_daily(self) -> None:
        """Reset daily P&L counter and clear circuit breaker (if applicable).

        PAUSED and SHUTDOWN states are NOT auto-cleared — those require
        explicit human intervention.
        """
        self.daily_pnl = 0.0
        if self.state == TradingState.CIRCUIT_BREAKER:
            self.state = TradingState.ACTIVE
            logger.info("Daily reset: circuit breaker cleared, state -> ACTIVE")
        else:
            logger.info("Daily reset: P&L zeroed, state remains %s", self.state.value)

    # ------------------------------------------------------------------
    # Manual controls
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Manually pause all trading. Requires explicit resume()."""
        self.state = TradingState.PAUSED
        logger.warning("Trading PAUSED by manual override")

    def resume(self) -> None:
        """Resume trading after a manual pause.

        Cannot resume from SHUTDOWN — that state is terminal until restart.
        """
        if self.state == TradingState.SHUTDOWN:
            logger.error("Cannot resume from SHUTDOWN state")
            return
        self.state = TradingState.ACTIVE
        logger.info("Trading RESUMED — state -> ACTIVE")

    # ------------------------------------------------------------------
    # Peak tracking
    # ------------------------------------------------------------------

    def update_peak(self, portfolio_value: float) -> None:
        """Update the high-water mark if portfolio_value is a new peak."""
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

    # ------------------------------------------------------------------
    # Status snapshot
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a complete snapshot of risk manager state and rules."""
        drawdown = self.check_drawdown(self.peak_value)  # drawdown vs current peak context
        return {
            "state": self.state.value,
            "daily_pnl": self.daily_pnl,
            "peak_value": self.peak_value,
            "starting_capital": self.starting_capital,
            "drawdown": drawdown,
            "rules": dict(self.IMMUTABLE_RULES),
        }
