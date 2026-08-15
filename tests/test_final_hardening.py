import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.routes.health import health_check, readiness_check, version_info
from app.config.settings import Settings, settings
from app.config.validation import ConfigurationError, validate_production_settings
from app.main import app
from app.policies.store_policy import ActionRequirement, get_action_reason, get_action_requirement
from app.schemas.monitoring import LocationContext, MonitoringRequest
from app.schemas.pricing import PricingBatchLLMResult, PricingBatchRequest, PricingDecision
from app.schemas.store_policy import OperatingMode, StorePolicy
from app.vector_store import VectorStoreError, get_vector_store, set_vector_store
from tests.test_pricing_agent import create_sample_batch_request

client = TestClient(app)


def test_operating_mode_semantics():
    """Requirement 27.1 & 27.2: Verify assisted and autonomous modes map to action requirements and reasons."""
    pol_ast = StorePolicy(store_id="s1", operating_mode=OperatingMode.ASSISTED)
    pol_aut = StorePolicy(store_id="s1", operating_mode=OperatingMode.AUTONOMOUS)

    assert get_action_requirement(pol_ast) == ActionRequirement.APPROVAL_REQUIRED
    assert get_action_requirement(pol_aut) == ActionRequirement.AUTOMATIC_EXECUTION_ELIGIBLE

    reason_ast = get_action_reason(pol_ast)
    reason_aut = get_action_reason(pol_aut)

    assert "assisted mode" in reason_ast
    assert "owner approval is required" in reason_ast
    assert "autonomous mode" in reason_aut
    assert "automatic execution" in reason_aut


def test_api_health_ready_version_endpoints():
    """Requirement 27.4: Test /health, /ready, and /version API endpoints."""
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "ok"}

    res_ready = client.get("/ready")
    assert res_ready.status_code == 200
    assert res_ready.json()["status"] == "ready"
    assert "configuration" in res_ready.json()["checks"]

    res_ver = client.get("/version")
    assert res_ver.status_code == 200
    data_ver = res_ver.json()
    assert data_ver["app_name"] == settings.APP_NAME
    assert data_ver["version"] == settings.APP_VERSION
    assert "environment" in data_ver


def test_correlation_id_middleware():
    """Requirement 27.5: Test correlation ID preservation and generation."""
    custom_id = "test-corr-id-12345"
    resp_custom = client.get("/health", headers={"X-Request-ID": custom_id})
    assert resp_custom.status_code == 200
    assert resp_custom.headers.get("X-Request-ID") == custom_id

    resp_auto = client.get("/health")
    assert resp_auto.status_code == 200
    assert "X-Request-ID" in resp_auto.headers
    assert len(resp_auto.headers["X-Request-ID"]) > 0


def test_batch_size_limit_validation():
    """Requirement 27.6: Test batch size exceeding MAX_PRICING_BATCH_SIZE is rejected."""
    base_req = create_sample_batch_request(store_id="s1", product_ids=["p1"])

    dup_products = [base_req.products[0], base_req.products[0]]
    with pytest.raises(ValueError) as exc_info:
        PricingBatchRequest(store_id="s1", products=dup_products)
    assert "Duplicate product_id" in str(exc_info.value)

    with patch.object(settings, "MAX_PRICING_BATCH_SIZE", 1):
        with pytest.raises(ValueError) as exc_info:
            PricingBatchRequest(store_id="s1", products=[base_req.products[0], base_req.products[0]])


def test_error_contract_validation():
    """Requirement 27.7: Test API error contract for invalid input validation."""
    response = client.post("/api/v1/pricing/recommend", json={"store_id": ""})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert "message" in body


def test_dockerfile_and_dockerignore_exist():
    """Requirement 27.10: Test Dockerfile and .dockerignore files exist and contain valid directives."""
    assert os.path.exists("Dockerfile")
    assert os.path.exists(".dockerignore")

    with open("Dockerfile", "r") as f:
        df_content = f.read()
    assert "FROM python:" in df_content
    assert "USER appuser" in df_content
    assert "--reload" not in df_content

    with open(".dockerignore", "r") as f:
        di_content = f.read()
    assert ".venv" in di_content
    assert ".env" in di_content


def test_production_settings_validation_rejects_fake_and_memory_in_production():
    """Requirement 8 & 9: Verify validate_production_settings rejects fake providers and memory vector store in production environment."""
    prod_settings = Settings(
        APP_ENV="production",
        OPENAI_API_KEY="sk-real-key-placeholder",
        OPENAI_MODEL="gemma-2-27b-it",
        EMBEDDING_PROVIDER="fake",
        VECTOR_STORE_PROVIDER="memory",
        WEATHER_PROVIDER="mock",
        EVENTS_PROVIDER="mock",
    )

    with pytest.raises(ConfigurationError) as exc_info:
        validate_production_settings(prod_settings)

    err_str = str(exc_info.value)
    assert "VECTOR_STORE_PROVIDER must be 'qdrant'" in err_str
    assert "EMBEDDING_PROVIDER must be 'local_bge_m3'" in err_str
    assert "WEATHER_PROVIDER must be 'open_meteo'" in err_str
    assert "EVENTS_PROVIDER must be 'nager_date' or 'nager'" in err_str


def test_no_silent_fallback_to_memory_when_qdrant_unreachable():
    """Requirement 2: Verify get_vector_store does not silently fall back to memory when Qdrant fails."""
    set_vector_store(None)
    with patch.object(settings, "VECTOR_STORE_PROVIDER", "qdrant"):
        with patch("app.vector_store.factory.QdrantVectorStore", side_effect=VectorStoreError("Failed to connect to Qdrant cluster")):
            with pytest.raises(VectorStoreError) as exc_info:
                get_vector_store()
            assert "Failed to connect to Qdrant cluster" in str(exc_info.value)
    set_vector_store(None)

