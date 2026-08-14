from enum import Enum
from pydantic import BaseModel

from app.schemas.monitoring import MonitoringRequest


class SignalLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskSignals(BaseModel):
    expiry_pressure: SignalLevel
    inventory_pressure: SignalLevel
    demand_pressure: SignalLevel


def calculate_expiry_pressure(hours_remaining: float) -> SignalLevel:
    """Calculates deterministic expiry pressure based on remaining shelf-life hours."""
    if hours_remaining <= 24:
        return SignalLevel.HIGH
    elif hours_remaining <= 48:
        return SignalLevel.MEDIUM
    else:
        return SignalLevel.LOW


def calculate_inventory_pressure(
    quantity: int, sales_velocity: float, hours_remaining: float
) -> SignalLevel:
    """Calculates deterministic inventory pressure based on inventory ratio to expected near-term sales."""
    days_remaining = hours_remaining / 24.0
    expected_sales = sales_velocity * days_remaining
    inventory_ratio = quantity / max(expected_sales, 1.0)

    if inventory_ratio <= 1.0:
        return SignalLevel.LOW
    elif inventory_ratio <= 2.0:
        return SignalLevel.MEDIUM
    else:
        return SignalLevel.HIGH


def calculate_demand_pressure(
    sales_velocity: float, historical_average_daily_sales: float
) -> SignalLevel:
    """Calculates deterministic demand pressure based on sales velocity ratio to historical average daily sales."""
    demand_ratio = sales_velocity / max(historical_average_daily_sales, 1.0)

    if demand_ratio >= 1.0:
        return SignalLevel.LOW
    elif demand_ratio >= 0.5:
        return SignalLevel.MEDIUM
    else:
        return SignalLevel.HIGH


def calculate_risk_signals(request: MonitoringRequest) -> RiskSignals:
    """Calculates all deterministic risk signals from a MonitoringRequest."""
    expiry_press = calculate_expiry_pressure(request.expiry.hours_remaining)
    inventory_press = calculate_inventory_pressure(
        quantity=request.inventory.quantity,
        sales_velocity=request.demand.sales_velocity,
        hours_remaining=request.expiry.hours_remaining,
    )
    demand_press = calculate_demand_pressure(
        sales_velocity=request.demand.sales_velocity,
        historical_average_daily_sales=request.demand.historical_sales.average_daily_sales,
    )

    return RiskSignals(
        expiry_pressure=expiry_press,
        inventory_pressure=inventory_press,
        demand_pressure=demand_press,
    )
