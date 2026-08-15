"""Centralized business configuration and thresholds for the Pricing Agent decision framework."""

# Maximum allowed AI recommendation discount percentage
MAX_AI_DISCOUNT: float = 15.0
MIN_AI_DISCOUNT: float = 0.0

# Expiry pressure threshold boundaries (in hours remaining)
EXPIRY_CRITICAL_THRESHOLD_HOURS: float = 24.0
EXPIRY_HIGH_THRESHOLD_HOURS: float = 48.0
EXPIRY_MODERATE_THRESHOLD_HOURS: float = 72.0

# Inventory coverage threshold boundaries (in days)
INVENTORY_LOW_COVERAGE_DAYS: float = 1.0
INVENTORY_MODERATE_COVERAGE_DAYS: float = 3.0
INVENTORY_HIGH_COVERAGE_DAYS: float = 7.0

# Demand ratio threshold boundaries (sales_velocity / historical_average_daily_sales)
DEMAND_STRONG_RATIO: float = 1.2
DEMAND_NORMAL_RATIO: float = 0.8
DEMAND_WEAK_RATIO: float = 0.5
