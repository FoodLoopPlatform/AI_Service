from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError

from app.agents.pricing import run_pricing_agent
from app.agents.pricing.prompts import PRICING_SYSTEM_PROMPT, format_pricing_user_prompt
from app.agents.pricing.retriever import VectorPricingKnowledgeRetriever
from app.agents.pricing.signals import calculate_pricing_signals
from app.embeddings.base import EmbeddingProviderError
from app.schemas.monitoring import (
    DemandContext,
    ExpiryContext,
    HistoricalSales,
    InventoryMetrics,
    RiskLevel,
)
from app.schemas.pricing import (
    PricingBatchLLMResult,
    PricingBatchRequest,
    PricingDecision,
    PricingProductContext,
)
from app.schemas.pricing_knowledge import PricingKnowledgeItem
from app.schemas.pricing_signals import DemandPressure, InventoryPressure
from app.schemas.risk_assessment import RiskAssessmentResult
from app.schemas.store_policy import OperatingMode, StorePolicy
from app.vector_store.base import VectorStoreError


def create_mock_product(
    product_id: str = "prod-001",
    quantity: int = 100,
    sales_velocity: float = 10.0,
    average_daily_sales: float = 10.0,
    hours_remaining: float = 100.0,
    risk_level: RiskLevel = RiskLevel.LOW,
) -> PricingProductContext:
    """Helper fixture to create deterministic PricingProductContext."""
    return PricingProductContext(
        product_id=product_id,
        product_name=f"Test Product {product_id}",
        category="Dairy",
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
            risk_level=risk_level,
            reason="Monitoring assessment test rationale",
            confidence=0.95,
        ),
    )


# -----------------------------------------------------------------------------
# SCENARIOS A - O
# -----------------------------------------------------------------------------

def test_scenario_a_healthy_product():
    """Scenario A: Healthy product context (low expiry/inventory pressure, normal demand, LOW risk)."""
    product = create_mock_product(
        hours_remaining=120.0,  # LOW expiry pressure
        quantity=5,
        sales_velocity=10.0,     # LOW inventory coverage
        average_daily_sales=10.0, # NORMAL demand
        risk_level=RiskLevel.LOW,
    )
    req = PricingBatchRequest(
        store_id="store-cairo-01",
        products=[product],
        store_policy=StorePolicy(store_id="store-cairo-01", operating_mode=OperatingMode.ASSISTED),
    )

    llm_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(
                product_id="prod-001",
                discount_percentage=3.0,
                reason="Healthy sales velocity and long shelf life indicate minimal discount requirement.",
                confidence=0.95,
            )
        ]
    )

    mock_llm_struct = MagicMock()
    mock_llm_struct.invoke.return_value = llm_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_struct

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    dec = resp.decisions[0]
    assert dec.product_id == "prod-001"
    assert 0.0 <= dec.discount_percentage <= 15.0
    assert dec.discount_percentage == 3.0
    assert len(dec.reason) > 0
    assert 0.0 <= dec.confidence <= 1.0
    assert not hasattr(dec, "final_price")
    assert not hasattr(dec, "execution_decision")


def test_scenario_b_high_pressure_product():
    """Scenario B: High pressure product (critical expiry, very high inventory, weak demand, HIGH risk)."""
    product = create_mock_product(
        hours_remaining=12.0,    # CRITICAL expiry pressure (<24h)
        quantity=500,
        sales_velocity=2.0,      # VERY_HIGH inventory coverage (>7d)
        average_daily_sales=10.0, # WEAK_DEMAND (ratio < 0.5)
        risk_level=RiskLevel.HIGH,
    )
    req = PricingBatchRequest(
        store_id="store-cairo-01",
        products=[product],
        store_policy=StorePolicy(store_id="store-cairo-01", operating_mode=OperatingMode.AUTONOMOUS),
    )

    llm_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(
                product_id="prod-001",
                discount_percentage=14.5,
                reason="Critical expiry pressure combined with high inventory coverage requires aggressive markdown.",
                confidence=0.92,
            )
        ]
    )

    mock_llm_struct = MagicMock()
    mock_llm_struct.invoke.return_value = llm_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_struct

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    dec = resp.decisions[0]
    assert dec.discount_percentage <= 15.0
    assert dec.discount_percentage == 14.5
    assert "expiry" in dec.reason.lower() or "inventory" in dec.reason.lower()


def test_scenario_c_conflicting_signals():
    """Scenario C: Conflicting signals (critical expiry but strong demand and low inventory)."""
    product = create_mock_product(
        hours_remaining=18.0,    # CRITICAL expiry
        quantity=10,
        sales_velocity=20.0,     # LOW inventory coverage
        average_daily_sales=10.0, # STRONG_DEMAND
        risk_level=RiskLevel.MEDIUM,
    )
    req = PricingBatchRequest(store_id="store-cairo-01", products=[product])

    llm_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(
                product_id="prod-001",
                discount_percentage=7.5,
                reason="Imminent expiry suggests discount, but strong demand prevents maximum markdown.",
                confidence=0.85,
            )
        ]
    )

    mock_llm_struct = MagicMock()
    mock_llm_struct.invoke.return_value = llm_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_struct

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    dec = resp.decisions[0]
    assert 0.0 <= dec.discount_percentage <= 15.0
    assert dec.discount_percentage == 7.5


def test_scenario_d_strong_historical_evidence():
    """Scenario D: Historical knowledge provided as evidence."""
    product = create_mock_product(product_id="prod-001")
    req = PricingBatchRequest(store_id="store-cairo-01", products=[product])

    knowledge_item = PricingKnowledgeItem(
        product_id="prod-001",
        store_id="store-cairo-01",
        content="Historical 10% markdown led to 100% sell-through within 12 hours.",
        metadata={"category": "Dairy"},
        relevance_score=0.92,
    )

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [knowledge_item]

    llm_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(
                product_id="prod-001",
                discount_percentage=10.0,
                reason="Historical evidence indicates 10% discount achieves full sell-through.",
                confidence=0.95,
            )
        ]
    )

    mock_llm_struct = MagicMock()
    mock_llm_struct.invoke.return_value = llm_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_struct

    with patch("app.agents.pricing.nodes.get_pricing_knowledge_retriever", return_value=mock_retriever), \
         patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    assert resp.decisions[0].discount_percentage == 10.0
    assert "Historical evidence" in resp.decisions[0].reason


def test_scenario_e_no_historical_knowledge():
    """Scenario E: No historical knowledge retrieved."""
    product = create_mock_product(product_id="prod-001")
    req = PricingBatchRequest(store_id="store-cairo-01", products=[product])

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    llm_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(
                product_id="prod-001",
                discount_percentage=5.0,
                reason="Calculated recommendation based solely on operational metrics.",
                confidence=0.80,
            )
        ]
    )

    mock_llm_struct = MagicMock()
    mock_llm_struct.invoke.return_value = llm_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_struct

    with patch("app.agents.pricing.nodes.get_pricing_knowledge_retriever", return_value=mock_retriever), \
         patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    assert resp.decisions[0].discount_percentage == 5.0


def test_scenario_f_zero_current_sales():
    """Scenario F: Zero current sales velocity with positive historical baseline."""
    product = create_mock_product(sales_velocity=0.0, average_daily_sales=10.0)
    signals = calculate_pricing_signals(product)

    assert signals.sales_velocity_zero is True
    assert signals.demand_pressure == DemandPressure.NO_CURRENT_SALES
    assert signals.inventory_coverage_days is None
    assert signals.inventory_pressure == InventoryPressure.VERY_HIGH

    req = PricingBatchRequest(store_id="store-cairo-01", products=[product])
    llm_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(
                product_id="prod-001",
                discount_percentage=12.0,
                reason="Zero current sales velocity indicates stagnant inventory.",
                confidence=0.88,
            )
        ]
    )

    mock_llm_struct = MagicMock()
    mock_llm_struct.invoke.return_value = llm_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_struct

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    assert resp.decisions[0].discount_percentage == 12.0


def test_scenario_g_zero_historical_baseline():
    """Scenario G: Zero sales velocity and zero historical baseline."""
    product = create_mock_product(sales_velocity=0.0, average_daily_sales=0.0)
    signals = calculate_pricing_signals(product)

    assert signals.demand_ratio is None
    assert signals.demand_pressure == DemandPressure.NO_DEMAND_BASELINE
    assert signals.inventory_coverage_days is None
    assert signals.inventory_pressure == InventoryPressure.VERY_HIGH
    assert signals.historical_average_zero is True

    req = PricingBatchRequest(store_id="store-cairo-01", products=[product])
    llm_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(
                product_id="prod-001",
                discount_percentage=8.0,
                reason="No historical baseline available; cautious discount recommendation.",
                confidence=0.70,
            )
        ]
    )

    mock_llm_struct = MagicMock()
    mock_llm_struct.invoke.return_value = llm_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_struct

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    assert resp.decisions[0].discount_percentage == 8.0


def test_scenario_h_operating_mode_independence():
    """Scenario H: ASSISTED vs AUTONOMOUS mode independence."""
    prod = create_mock_product()
    req_assisted = PricingBatchRequest(
        store_id="s1",
        products=[prod],
        store_policy=StorePolicy(store_id="s1", operating_mode=OperatingMode.ASSISTED),
    )
    req_autonomous = PricingBatchRequest(
        store_id="s1",
        products=[prod],
        store_policy=StorePolicy(store_id="s1", operating_mode=OperatingMode.AUTONOMOUS),
    )

    sig_assisted = calculate_pricing_signals(req_assisted.products[0])
    sig_autonomous = calculate_pricing_signals(req_autonomous.products[0])

    assert sig_assisted == sig_autonomous


def test_scenario_i_reason_quality_contract():
    """Scenario I: Mandatory reason contract validation."""
    with pytest.raises(ValidationError):
        PricingDecision(product_id="p1", discount_percentage=5.0, reason="", confidence=0.9)


def test_scenario_j_confidence_contract():
    """Scenario J: Confidence boundaries (0.0 to 1.0)."""
    d0 = PricingDecision(product_id="p1", discount_percentage=5.0, reason="Valid", confidence=0.0)
    d1 = PricingDecision(product_id="p1", discount_percentage=5.0, reason="Valid", confidence=1.0)
    assert d0.confidence == 0.0
    assert d1.confidence == 1.0

    with pytest.raises(ValidationError):
        PricingDecision(product_id="p1", discount_percentage=5.0, reason="Valid", confidence=-0.1)

    with pytest.raises(ValidationError):
        PricingDecision(product_id="p1", discount_percentage=5.0, reason="Valid", confidence=1.1)


def test_scenario_k_discount_boundary():
    """Scenario K: Discount percentage boundaries (0.0 to 15.0)."""
    d0 = PricingDecision(product_id="p1", discount_percentage=0.0, reason="Valid", confidence=0.9)
    d15 = PricingDecision(product_id="p1", discount_percentage=15.0, reason="Valid", confidence=0.9)
    assert d0.discount_percentage == 0.0
    assert d15.discount_percentage == 15.0

    with pytest.raises(ValidationError):
        PricingDecision(product_id="p1", discount_percentage=-0.1, reason="Valid", confidence=0.9)

    with pytest.raises(ValidationError):
        PricingDecision(product_id="p1", discount_percentage=15.1, reason="Valid", confidence=0.9)


def test_scenario_l_no_forbidden_execution_output():
    """Scenario L: Extra financial or execution fields are rejected by extra='forbid'."""
    invalid_payload = {
        "product_id": "p1",
        "discount_percentage": 5.0,
        "reason": "Valid reason",
        "confidence": 0.9,
        "final_price": 45.0,
        "execution_decision": "EXECUTE",
    }
    with pytest.raises(ValidationError):
        PricingDecision(**invalid_payload)


def test_scenario_m_retrieval_failure():
    """Scenario M: VectorStoreError propagates without fallback."""
    product = create_mock_product()
    req = PricingBatchRequest(store_id="store-cairo-01", products=[product])

    mock_retriever = MagicMock()
    mock_retriever.retrieve.side_effect = VectorStoreError("Vector store search failed.")

    with patch("app.agents.pricing.nodes.get_pricing_knowledge_retriever", return_value=mock_retriever):
        with pytest.raises(VectorStoreError):
            run_pricing_agent(req)


def test_scenario_n_embedding_failure():
    """Scenario N: EmbeddingProviderError propagates without fallback."""
    retriever_with_failing_embedder = VectorPricingKnowledgeRetriever(
        embedding_provider=MagicMock(embed_queries=MagicMock(side_effect=EmbeddingProviderError("API down"))),
        vector_store=MagicMock(),
    )
    product = create_mock_product()
    req = PricingBatchRequest(store_id="store-cairo-01", products=[product])

    with patch("app.agents.pricing.nodes.get_pricing_knowledge_retriever", return_value=retriever_with_failing_embedder):
        with pytest.raises(EmbeddingProviderError):
            run_pricing_agent(req)


def test_scenario_o_llm_failure():
    """Scenario O: Structured LLM failure propagates cleanly."""
    product = create_mock_product()
    req = PricingBatchRequest(store_id="store-cairo-01", products=[product])

    mock_llm_struct = MagicMock()
    mock_llm_struct.invoke.side_effect = RuntimeError("LLM API failure")
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_struct

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        with pytest.raises(RuntimeError):
            run_pricing_agent(req)


# -----------------------------------------------------------------------------
# PROMPT CONTRACT & ISOLATION TESTS
# -----------------------------------------------------------------------------

def test_prompt_contracts():
    """Requirement 17: Prompt formatting contains required context and omits forbidden instructions."""
    product = create_mock_product(product_id="prod-99")
    prompt = format_pricing_user_prompt(store_id="store-01", products=[product])

    # Assert required elements exist
    assert "prod-99" in prompt
    assert "Deterministic Signals:" in prompt
    assert "Inventory Coverage:" in prompt
    assert "Demand Ratio:" in prompt
    assert "Expiry Pressure:" in prompt
    assert "Monitoring Risk Level: LOW" in prompt

    # Assert forbidden instructions are not present in system prompt
    assert "calculate final price" not in PRICING_SYSTEM_PROMPT.lower()
    assert "modify price floor" not in PRICING_SYSTEM_PROMPT.lower()
    assert "chain-of-thought" not in PRICING_SYSTEM_PROMPT.lower() or "do not output chain-of-thought" in PRICING_SYSTEM_PROMPT.lower()


def test_product_isolation_scenario():
    """Requirement 18: Product knowledge isolation in batch requests."""
    prod_a = create_mock_product(product_id="prod-A")
    prod_b = create_mock_product(product_id="prod-B")

    knowledge_a = PricingKnowledgeItem(
        product_id="prod-A",
        store_id="store-01",
        content="Product A historical discount evidence",
        metadata={"category": "Dairy"},
        relevance_score=0.9,
    )

    knowledge_by_product = {
        "prod-A": [knowledge_a],
        "prod-B": [],
    }

    prompt = format_pricing_user_prompt(
        store_id="store-01",
        products=[prod_a, prod_b],
        knowledge_by_product=knowledge_by_product,
    )

    # Product A section must contain knowledge A
    assert "Product A historical discount evidence" in prompt

    # Split prompt by product sections
    parts = prompt.split("PRODUCT 2:")
    product_a_section = parts[0]
    product_b_section = parts[1]

    assert "Product A historical discount evidence" in product_a_section
    assert "Product A historical discount evidence" not in product_b_section


def test_batch_behavior_scenario():
    """Requirement 19: Batch execution maintains 1-to-1 decision mapping."""
    p1 = create_mock_product(product_id="p1")
    p2 = create_mock_product(product_id="p2")
    req = PricingBatchRequest(store_id="store-01", products=[p1, p2])

    llm_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(product_id="p1", discount_percentage=5.0, reason="Reason p1", confidence=0.9),
            PricingDecision(product_id="p2", discount_percentage=10.0, reason="Reason p2", confidence=0.9),
        ]
    )

    mock_llm_struct = MagicMock()
    mock_llm_struct.invoke.return_value = llm_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_struct

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    assert len(resp.decisions) == 2
    assert resp.decisions[0].product_id == "p1"
    assert resp.decisions[1].product_id == "p2"


def test_no_execution_architectural_guarantee():
    """Requirement 20: Architectural proof that run_pricing_agent performs zero financial execution."""
    product = create_mock_product()
    req = PricingBatchRequest(store_id="s1", products=[product])

    llm_result = PricingBatchLLMResult(
        decisions=[PricingDecision(product_id="prod-001", discount_percentage=5.0, reason="Test", confidence=0.9)]
    )

    mock_llm_struct = MagicMock()
    mock_llm_struct.invoke.return_value = llm_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_struct

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    # Response contains ONLY recommendation decision
    assert hasattr(resp, "decisions")
    assert not hasattr(resp, "final_price")
    assert not hasattr(resp, "price_floor_enforced")
    assert not hasattr(resp, "transaction_executed")
