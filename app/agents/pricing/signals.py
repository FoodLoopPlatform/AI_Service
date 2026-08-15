from app.agents.pricing.config import (
    DEMAND_NORMAL_RATIO,
    DEMAND_STRONG_RATIO,
    DEMAND_WEAK_RATIO,
    EXPIRY_CRITICAL_THRESHOLD_HOURS,
    EXPIRY_HIGH_THRESHOLD_HOURS,
    EXPIRY_MODERATE_THRESHOLD_HOURS,
    INVENTORY_HIGH_COVERAGE_DAYS,
    INVENTORY_LOW_COVERAGE_DAYS,
    INVENTORY_MODERATE_COVERAGE_DAYS,
)
from app.schemas.pricing import PricingProductContext
from app.schemas.pricing_signals import (
    DemandPressure,
    ExpiryPressure,
    InventoryPressure,
    PricingSignals,
)


def calculate_pricing_signals(product: PricingProductContext) -> PricingSignals:
    """Calculates deterministic pricing signals from a product's operational context.
    
    Zero LLM calls are involved. Thresholds are derived from centralized configuration.
    """
    quantity = float(product.inventory.quantity)
    velocity = float(product.demand.sales_velocity)

    # 1. Inventory Coverage & Pressure
    sales_velocity_zero = (velocity == 0.0)
    if sales_velocity_zero:
        inventory_coverage_days = None
        inventory_pressure = InventoryPressure.VERY_HIGH
    else:
        inventory_coverage_days = round(quantity / velocity, 4)
        if inventory_coverage_days <= INVENTORY_LOW_COVERAGE_DAYS:
            inventory_pressure = InventoryPressure.LOW
        elif inventory_coverage_days <= INVENTORY_MODERATE_COVERAGE_DAYS:
            inventory_pressure = InventoryPressure.MODERATE
        elif inventory_coverage_days <= INVENTORY_HIGH_COVERAGE_DAYS:
            inventory_pressure = InventoryPressure.HIGH
        else:
            inventory_pressure = InventoryPressure.VERY_HIGH

    # 2. Demand Ratio & Demand Pressure
    if hasattr(product.demand, "historical_sales") and hasattr(product.demand.historical_sales, "average_daily_sales"):
        hist_avg = float(product.demand.historical_sales.average_daily_sales)
    else:
        hist_avg = float(getattr(product.demand, "historical_average_daily_sales", 0.0))

    historical_average_zero = (hist_avg == 0.0)

    if sales_velocity_zero and historical_average_zero:
        demand_ratio = None
        demand_pressure = DemandPressure.NO_DEMAND_BASELINE
    elif sales_velocity_zero and not historical_average_zero:
        demand_ratio = 0.0
        demand_pressure = DemandPressure.NO_CURRENT_SALES
    elif historical_average_zero:
        demand_ratio = None
        demand_pressure = DemandPressure.NO_DEMAND_BASELINE
    else:
        demand_ratio = round(velocity / hist_avg, 4)
        if demand_ratio >= DEMAND_STRONG_RATIO:
            demand_pressure = DemandPressure.STRONG_DEMAND
        elif demand_ratio >= DEMAND_NORMAL_RATIO:
            demand_pressure = DemandPressure.NORMAL_DEMAND
        elif demand_ratio >= DEMAND_WEAK_RATIO:
            demand_pressure = DemandPressure.WEAK_DEMAND
        else:
            demand_pressure = DemandPressure.VERY_WEAK_DEMAND

    # 3. Expiry Pressure
    hours = float(product.expiry.hours_remaining)
    if hours < EXPIRY_CRITICAL_THRESHOLD_HOURS:
        expiry_pressure = ExpiryPressure.CRITICAL
    elif hours < EXPIRY_HIGH_THRESHOLD_HOURS:
        expiry_pressure = ExpiryPressure.HIGH
    elif hours <= EXPIRY_MODERATE_THRESHOLD_HOURS:
        expiry_pressure = ExpiryPressure.MODERATE
    else:
        expiry_pressure = ExpiryPressure.LOW

    return PricingSignals(
        inventory_coverage_days=inventory_coverage_days,
        inventory_pressure=inventory_pressure,
        demand_ratio=demand_ratio,
        demand_pressure=demand_pressure,
        expiry_pressure=expiry_pressure,
        sales_velocity_zero=sales_velocity_zero,
        historical_average_zero=historical_average_zero,
    )
