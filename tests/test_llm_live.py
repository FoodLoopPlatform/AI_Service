import os
import time
import pytest
from pydantic import BaseModel, Field

from app.config.settings import settings
from app.llm.factory import get_llm


def _should_skip_live_tests() -> bool:
    is_opt_in = os.getenv("RUN_EXTERNAL_INTEGRATION_TESTS") == "true"
    api_key = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
    has_key = bool(api_key and api_key.strip() and api_key != "placeholder-api-key")
    return not (is_opt_in and has_key)


class SmokeTestResult(BaseModel):
    """Tiny temporary schema for structured output verification."""

    answer: str = Field(description="Direct concise answer to the question.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")


@pytest.mark.skipif(
    _should_skip_live_tests(),
    reason="Opt-in live LLM test requires RUN_EXTERNAL_INTEGRATION_TESTS=true and OPENAI_API_KEY set",
)
def test_real_llm_live_connectivity_and_structured_output():
    """Opt-in live integration test verifying connectivity, base_url, model, and structured output."""
    base_url = settings.OPENAI_BASE_URL
    model_name = settings.OPENAI_MODEL

    # 1. Basic Connectivity
    start_time = time.perf_counter()
    llm = get_llm()

    # Verify model parameter on configured model instance
    configured_model = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    assert configured_model == model_name, f"Expected model '{model_name}', got '{configured_model}'"

    greeting_resp = llm.invoke("Return a short greeting.")
    greeting_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    assert greeting_resp is not None
    greeting_text = str(greeting_resp.content).strip()
    assert len(greeting_text) > 0, "Greeting response was empty"

    # 2. Structured Output Verification
    struct_start_time = time.perf_counter()
    structured_llm = llm.with_structured_output(SmokeTestResult)

    result = structured_llm.invoke(
        "What is 2 + 2? Return the direct numerical answer in 'answer' and a confidence score between 0.0 and 1.0 in 'confidence'."
    )
    struct_latency_ms = round((time.perf_counter() - struct_start_time) * 1000, 2)

    assert isinstance(result, SmokeTestResult), f"Expected SmokeTestResult, got {type(result)}"
    assert result.answer is not None and len(result.answer.strip()) > 0, "Structured output answer was empty"
    assert isinstance(result.confidence, float), f"Expected float confidence, got {type(result.confidence)}"
    assert 0.0 <= result.confidence <= 1.0, f"Confidence out of range [0, 1]: {result.confidence}"
