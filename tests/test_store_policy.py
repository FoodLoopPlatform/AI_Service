from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError

from app.agents.pricing import run_pricing_agent
from app.policies.store_policy import ActionRequirement, get_action_requirement
from app.schemas.monitoring import LocationContext, MonitoringRequest
from app.schemas.pricing import (
    PricingBatchLLMResult,
    PricingBatchRequest,
    PricingDecision,
)
from app.schemas.store_policy import OperatingMode, StorePolicy
from tests.test_monitoring_schema import get_valid_request_dict
from tests.test_pricing_agent import create_sample_batch_request


def test_store_policy_schema_valid_modes():
    """Requirement 14.A: assisted and autonomous are valid operating modes."""
    pol_ast = StorePolicy(store_id="store-01", operating_mode=OperatingMode.ASSISTED)
    pol_aut = StorePolicy(store_id="store-01", operating_mode=OperatingMode.AUTONOMOUS)

    assert pol_ast.operating_mode == "assisted"
    assert pol_aut.operating_mode == "autonomous"


def test_store_policy_obsolete_and_invalid_modes_rejected():
    """Requirement 14.A: recommendation, manual, and arbitrary strings are rejected."""
    for invalid_mode in ["recommendation", "manual", "fully_automated"]:
        with pytest.raises(ValidationError) as exc_info:
            StorePolicy(store_id="store-01", operating_mode=invalid_mode)  # type: ignore
        assert "operating_mode" in str(exc_info.value)


def test_store_policy_empty_store_id():
    """Requirement C: empty store_id is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        StorePolicy(store_id="", operating_mode=OperatingMode.ASSISTED)
    assert "store_id" in str(exc_info.value)


def test_store_policy_extra_fields_forbidden():
    """Requirement test: extra fields forbidden in StorePolicy."""
    with pytest.raises(ValidationError) as exc_info:
        StorePolicy(store_id="store-01", operating_mode=OperatingMode.ASSISTED, discount_cap=20.0)  # type: ignore
    assert "extra" in str(exc_info.value).lower() or "discount_cap" in str(exc_info.value)


def test_deterministic_policy_mapping():
    """Requirement 14.B & Requirement 4: Policy mapping translates assisted -> APPROVAL_REQUIRED and autonomous -> AUTOMATIC_EXECUTION_ELIGIBLE."""
    pol_ast = StorePolicy(store_id="s1", operating_mode=OperatingMode.ASSISTED)
    pol_aut = StorePolicy(store_id="s1", operating_mode=OperatingMode.AUTONOMOUS)

    assert get_action_requirement(pol_ast) == ActionRequirement.APPROVAL_REQUIRED
    assert get_action_requirement(pol_aut) == ActionRequirement.AUTOMATIC_EXECUTION_ELIGIBLE


def test_monitoring_request_store_policy_consistency():
    """Requirement C: MonitoringRequest validates matching store_id in location and store_policy."""
    payload = get_valid_request_dict()
    base_req = MonitoringRequest(**payload)

    valid_policy = StorePolicy(store_id=base_req.location.store_id, operating_mode=OperatingMode.ASSISTED)
    mon_req = MonitoringRequest(
        product=base_req.product,
        inventory=base_req.inventory,
        demand=base_req.demand,
        expiry=base_req.expiry,
        location=base_req.location,
        timestamp=base_req.timestamp,
        store_policy=valid_policy,
    )
    assert mon_req.store_policy.operating_mode == OperatingMode.ASSISTED

    invalid_policy = StorePolicy(store_id="mismatched-store-999", operating_mode=OperatingMode.ASSISTED)
    with pytest.raises(ValueError) as exc_info:
        MonitoringRequest(
            product=base_req.product,
            inventory=base_req.inventory,
            demand=base_req.demand,
            expiry=base_req.expiry,
            location=base_req.location,
            timestamp=base_req.timestamp,
            store_policy=invalid_policy,
        )
    assert "store_id mismatch" in str(exc_info.value)


def test_pricing_batch_request_store_policy_consistency():
    """Requirement C: PricingBatchRequest validates matching store_id in request and store_policy."""
    base_req = create_sample_batch_request(store_id="store-cairo-01", product_ids=["p-100"])

    valid_policy = StorePolicy(store_id="store-cairo-01", operating_mode=OperatingMode.AUTONOMOUS)
    prc_req = PricingBatchRequest(
        store_id=base_req.store_id,
        products=base_req.products,
        store_policy=valid_policy,
    )
    assert prc_req.store_policy.operating_mode == OperatingMode.AUTONOMOUS

    invalid_policy = StorePolicy(store_id="store-alex-02", operating_mode=OperatingMode.AUTONOMOUS)
    with pytest.raises(ValueError) as exc_info:
        PricingBatchRequest(
            store_id=base_req.store_id,
            products=base_req.products,
            store_policy=invalid_policy,
        )
    assert "store_id mismatch" in str(exc_info.value)


def test_recommendation_independence_across_operating_modes():
    """Requirement 14.E & 12: For identical product input, assisted and autonomous modes yield identical Pricing Agent outputs."""
    base_req = create_sample_batch_request(store_id="store-cairo-01", product_ids=["p-100"])

    assisted_policy = StorePolicy(store_id="store-cairo-01", operating_mode=OperatingMode.ASSISTED)
    autonomous_policy = StorePolicy(store_id="store-cairo-01", operating_mode=OperatingMode.AUTONOMOUS)

    req_assisted = PricingBatchRequest(
        store_id=base_req.store_id,
        products=base_req.products,
        store_policy=assisted_policy,
    )
    req_autonomous = PricingBatchRequest(
        store_id=base_req.store_id,
        products=base_req.products,
        store_policy=autonomous_policy,
    )

    expected_llm_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(
                product_id="p-100",
                discount_percentage=10.0,
                reason="Standard discount rationale",
                confidence=0.9,
            )
        ]
    )

    mock_llm_structured = MagicMock()
    mock_llm_structured.invoke.return_value = expected_llm_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_structured

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp_assisted = run_pricing_agent(req_assisted)

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp_autonomous = run_pricing_agent(req_autonomous)

    assert resp_assisted.decisions[0].discount_percentage == resp_autonomous.decisions[0].discount_percentage
    assert resp_assisted.decisions[0].reason == resp_autonomous.decisions[0].reason
    assert resp_assisted.decisions[0].confidence == resp_autonomous.decisions[0].confidence
