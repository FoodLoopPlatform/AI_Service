from app.agents.monitoring.risk_signals import (
    SignalLevel,
    calculate_demand_pressure,
    calculate_expiry_pressure,
    calculate_inventory_pressure,
    calculate_risk_signals,
)
from app.schemas.monitoring import MonitoringRequest
from tests.test_monitoring_schema import get_valid_request_dict


def test_expiry_pressure_thresholds():
    assert calculate_expiry_pressure(0.0) == SignalLevel.HIGH
    assert calculate_expiry_pressure(12.0) == SignalLevel.HIGH
    assert calculate_expiry_pressure(24.0) == SignalLevel.HIGH
    assert calculate_expiry_pressure(24.01) == SignalLevel.MEDIUM
    assert calculate_expiry_pressure(48.0) == SignalLevel.MEDIUM
    assert calculate_expiry_pressure(48.01) == SignalLevel.LOW
    assert calculate_expiry_pressure(120.0) == SignalLevel.LOW


def test_inventory_pressure_thresholds():
    # 24 hours remaining = 1 day remaining
    # quantity = 5, sales_velocity = 10 -> expected_sales = 10 -> ratio = 0.5 (< 1.0) -> LOW
    assert calculate_inventory_pressure(5, 10.0, 24.0) == SignalLevel.LOW

    # quantity = 10, sales_velocity = 10 -> expected_sales = 10 -> ratio = 1.0 (= 1.0) -> LOW
    assert calculate_inventory_pressure(10, 10.0, 24.0) == SignalLevel.LOW

    # quantity = 15, sales_velocity = 10 -> expected_sales = 10 -> ratio = 1.5 (> 1.0) -> MEDIUM
    assert calculate_inventory_pressure(15, 10.0, 24.0) == SignalLevel.MEDIUM

    # quantity = 20, sales_velocity = 10 -> expected_sales = 10 -> ratio = 2.0 (= 2.0) -> MEDIUM
    assert calculate_inventory_pressure(20, 10.0, 24.0) == SignalLevel.MEDIUM

    # quantity = 21, sales_velocity = 10 -> expected_sales = 10 -> ratio = 2.1 (> 2.0) -> HIGH
    assert calculate_inventory_pressure(21, 10.0, 24.0) == SignalLevel.HIGH


def test_inventory_pressure_zero_sales_velocity_safely():
    # expected_sales = 0 * 1 = 0 -> max(0, 1) = 1 -> ratio = 50 / 1 = 50 -> HIGH
    assert calculate_inventory_pressure(50, 0.0, 24.0) == SignalLevel.HIGH


def test_demand_pressure_thresholds():
    # velocity = 20, avg = 15 -> ratio = 1.33 (> 1.0) -> LOW
    assert calculate_demand_pressure(20.0, 15.0) == SignalLevel.LOW

    # velocity = 15, avg = 15 -> ratio = 1.0 (= 1.0) -> LOW
    assert calculate_demand_pressure(15.0, 15.0) == SignalLevel.LOW

    # velocity = 10, avg = 15 -> ratio = 0.67 (between 0.5 and 1.0) -> MEDIUM
    assert calculate_demand_pressure(10.0, 15.0) == SignalLevel.MEDIUM

    # velocity = 7.5, avg = 15 -> ratio = 0.5 (= 0.5) -> MEDIUM
    assert calculate_demand_pressure(7.5, 15.0) == SignalLevel.MEDIUM

    # velocity = 7.0, avg = 15 -> ratio = 0.466 (< 0.5) -> HIGH
    assert calculate_demand_pressure(7.0, 15.0) == SignalLevel.HIGH


def test_demand_pressure_zero_historical_sales_safely():
    # velocity = 5, avg = 0 -> max(0, 1) = 1 -> ratio = 5.0 -> LOW
    assert calculate_demand_pressure(5.0, 0.0) == SignalLevel.LOW
    # velocity = 0, avg = 0 -> ratio = 0 -> HIGH
    assert calculate_demand_pressure(0.0, 0.0) == SignalLevel.HIGH


def test_calculate_risk_signals_integration():
    payload = get_valid_request_dict()
    request = MonitoringRequest(**payload)

    signals = calculate_risk_signals(request)
    assert signals.expiry_pressure in [SignalLevel.LOW, SignalLevel.MEDIUM, SignalLevel.HIGH]
    assert signals.inventory_pressure in [SignalLevel.LOW, SignalLevel.MEDIUM, SignalLevel.HIGH]
    assert signals.demand_pressure in [SignalLevel.LOW, SignalLevel.MEDIUM, SignalLevel.HIGH]
