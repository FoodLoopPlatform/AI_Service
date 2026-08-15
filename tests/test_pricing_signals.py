from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from app.agents.pricing.signals import calculate_pricing_signals
from app.schemas.monitoring import (
    DemandContext,
    ExpiryContext,
    HistoricalSales,
    InventoryMetrics,
    RiskLevel,
)
from app.schemas.pricing import PricingDecision, PricingProductContext
from app.schemas.pricing_signals import (
    DemandPressure,
    ExpiryPressure,
    InventoryPressure,
)
from app.schemas.risk_assessment import RiskAssessmentResult


def make_product_context(
    quantity: int = 100,
    sales_velocity: float = 10.0,
    average_daily_sales: float = 10.0,
    hours_remaining: float = 50.0,
    product_id: str = "prod-test-01",
) -> PricingProductContext:
    """Helper to generate a deterministic product context for testing pricing signals."""
    return PricingProductContext(
        product_id=product_id,
        product_name="Test Item",
        category="Test Category",
        inventory=InventoryMetrics(
            quantity=quantity,
            original_price=100.0,
            current_price=90.0,
            price_floor=50.0,
        ),
        demand=DemandContext(
            sales_velocity=sales_velocity,
            historical_sales=HistoricalSales(average_daily_sales=average_daily_sales),
        ),
        expiry=ExpiryContext(
            expires_at=datetime.now(timezone.utc),
            hours_remaining=hours_remaining,
        ),
        risk_assessment=RiskAssessmentResult(
            risk_level=RiskLevel.MEDIUM,
            reason="Sample risk assessment",
            confidence=0.9,
        ),
    )


def test_inventory_coverage_calculation():
    """Requirement 18.A: Verify inventory coverage days calculation."""
    # quantity=200, velocity=200 -> coverage = 1 day (LOW)
    ctx1 = make_product_context(quantity=200, sales_velocity=200.0)
    sig1 = calculate_pricing_signals(ctx1)
    assert sig1.inventory_coverage_days == 1.0
    assert sig1.inventory_pressure == InventoryPressure.LOW

    # quantity=200, velocity=5 -> coverage = 40 days (VERY_HIGH)
    ctx2 = make_product_context(quantity=200, sales_velocity=5.0)
    sig2 = calculate_pricing_signals(ctx2)
    assert sig2.inventory_coverage_days == 40.0
    assert sig2.inventory_pressure == InventoryPressure.VERY_HIGH


def test_zero_velocity_handling():
    """Requirement 18.B: Verify zero velocity does not raise ZeroDivisionError."""
    ctx = make_product_context(quantity=100, sales_velocity=0.0, average_daily_sales=10.0)
    sig = calculate_pricing_signals(ctx)
    assert sig.inventory_coverage_days is None
    assert sig.inventory_pressure == InventoryPressure.VERY_HIGH
    assert sig.sales_velocity_zero is True
    assert sig.demand_ratio == 0.0
    assert sig.demand_pressure == DemandPressure.NO_CURRENT_SALES


def test_demand_ratio_calculation():
    """Requirement 18.C: Verify demand ratio calculation."""
    # velocity=6, historical_average=5 -> ratio = 1.2 (STRONG_DEMAND)
    ctx = make_product_context(sales_velocity=6.0, average_daily_sales=5.0)
    sig = calculate_pricing_signals(ctx)
    assert sig.demand_ratio == 1.2
    assert sig.demand_pressure == DemandPressure.STRONG_DEMAND


def test_zero_historical_average_handling():
    """Requirement 18.C: Verify zero historical average daily sales handling."""
    ctx = make_product_context(sales_velocity=0.0, average_daily_sales=0.0)
    sig = calculate_pricing_signals(ctx)
    assert sig.demand_ratio is None
    assert sig.historical_average_zero is True
    assert sig.demand_pressure == DemandPressure.NO_DEMAND_BASELINE


def test_expiry_pressure_boundaries():
    """Requirement 18.D: Verify exact expiry pressure classification boundaries."""
    # < 24 -> CRITICAL
    sig_0 = calculate_pricing_signals(make_product_context(hours_remaining=0.0))
    assert sig_0.expiry_pressure == ExpiryPressure.CRITICAL

    sig_23_9 = calculate_pricing_signals(make_product_context(hours_remaining=23.9))
    assert sig_23_9.expiry_pressure == ExpiryPressure.CRITICAL

    # 24 to < 48 -> HIGH
    sig_24 = calculate_pricing_signals(make_product_context(hours_remaining=24.0))
    assert sig_24.expiry_pressure == ExpiryPressure.HIGH

    sig_24_01 = calculate_pricing_signals(make_product_context(hours_remaining=24.01))
    assert sig_24_01.expiry_pressure == ExpiryPressure.HIGH

    sig_47_9 = calculate_pricing_signals(make_product_context(hours_remaining=47.9))
    assert sig_47_9.expiry_pressure == ExpiryPressure.HIGH

    # 48 to <= 72 -> MODERATE
    sig_48 = calculate_pricing_signals(make_product_context(hours_remaining=48.0))
    assert sig_48.expiry_pressure == ExpiryPressure.MODERATE

    sig_48_01 = calculate_pricing_signals(make_product_context(hours_remaining=48.01))
    assert sig_48_01.expiry_pressure == ExpiryPressure.MODERATE

    sig_72 = calculate_pricing_signals(make_product_context(hours_remaining=72.0))
    assert sig_72.expiry_pressure == ExpiryPressure.MODERATE

    # > 72 -> LOW
    sig_72_01 = calculate_pricing_signals(make_product_context(hours_remaining=72.01))
    assert sig_72_01.expiry_pressure == ExpiryPressure.LOW


def test_inventory_pressure_boundaries():
    """Requirement 18.E: Verify exact inventory pressure classification boundaries."""
    # <= 1 day -> LOW
    sig_1 = calculate_pricing_signals(make_product_context(quantity=10, sales_velocity=10.0))
    assert sig_1.inventory_pressure == InventoryPressure.LOW

    # > 1 and <= 3 days -> MODERATE
    sig_3 = calculate_pricing_signals(make_product_context(quantity=30, sales_velocity=10.0))
    assert sig_3.inventory_pressure == InventoryPressure.MODERATE

    # > 3 and <= 7 days -> HIGH
    sig_7 = calculate_pricing_signals(make_product_context(quantity=70, sales_velocity=10.0))
    assert sig_7.inventory_pressure == InventoryPressure.HIGH

    # > 7 days -> VERY_HIGH
    sig_7_1 = calculate_pricing_signals(make_product_context(quantity=71, sales_velocity=10.0))
    assert sig_7_1.inventory_pressure == InventoryPressure.VERY_HIGH


def test_demand_pressure_boundaries():
    """Requirement 18.F: Verify exact demand pressure classification boundaries."""
    # demand_ratio >= 1.2 -> STRONG_DEMAND
    sig_1_2 = calculate_pricing_signals(make_product_context(sales_velocity=12.0, average_daily_sales=10.0))
    assert sig_1_2.demand_pressure == DemandPressure.STRONG_DEMAND

    # 0.8 <= ratio < 1.2 -> NORMAL_DEMAND
    sig_1_0 = calculate_pricing_signals(make_product_context(sales_velocity=10.0, average_daily_sales=10.0))
    assert sig_1_0.demand_pressure == DemandPressure.NORMAL_DEMAND

    sig_0_8 = calculate_pricing_signals(make_product_context(sales_velocity=8.0, average_daily_sales=10.0))
    assert sig_0_8.demand_pressure == DemandPressure.NORMAL_DEMAND

    # 0.5 <= ratio < 0.8 -> WEAK_DEMAND
    sig_0_5 = calculate_pricing_signals(make_product_context(sales_velocity=5.0, average_daily_sales=10.0))
    assert sig_0_5.demand_pressure == DemandPressure.WEAK_DEMAND

    # ratio < 0.5 -> VERY_WEAK_DEMAND
    sig_0_4 = calculate_pricing_signals(make_product_context(sales_velocity=4.0, average_daily_sales=10.0))
    assert sig_0_4.demand_pressure == DemandPressure.VERY_WEAK_DEMAND


def test_pricing_decision_output_constraints():
    """Requirement 18.G: Verify pricing decision schema bounds and mandatory fields."""
    # 0 accepted
    d0 = PricingDecision(product_id="p1", discount_percentage=0.0, reason="No discount needed", confidence=0.9)
    assert d0.discount_percentage == 0.0

    # 15 accepted
    d15 = PricingDecision(product_id="p1", discount_percentage=15.0, reason="Maximum discount applied", confidence=0.9)
    assert d15.discount_percentage == 15.0

    # negative discount rejected
    with pytest.raises(ValidationError):
        PricingDecision(product_id="p1", discount_percentage=-1.0, reason="Invalid", confidence=0.9)

    # > 15 discount rejected
    with pytest.raises(ValidationError):
        PricingDecision(product_id="p1", discount_percentage=15.1, reason="Invalid", confidence=0.9)

    # empty reason rejected
    with pytest.raises(ValidationError):
        PricingDecision(product_id="p1", discount_percentage=10.0, reason="", confidence=0.9)

    # invalid confidence rejected
    with pytest.raises(ValidationError):
        PricingDecision(product_id="p1", discount_percentage=10.0, reason="Valid reason", confidence=1.5)
