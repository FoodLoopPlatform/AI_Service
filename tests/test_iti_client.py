from unittest.mock import patch, MagicMock
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from app.llm.iti import ChatITI
from app.schemas.pricing import PricingBatchLLMResult


class DummySchema(BaseModel):
    summary: str = Field(description="A brief summary")
    score: float = Field(description="A score between 0 and 1")


def test_chat_iti_generate():
    client = ChatITI(
        model_name="google.gemma-3-27b-it",
        api_key="test-key",
        base_url="http://apiaccess.iti.net.eg/api/v1",
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "request_id": "test-id",
        "model_id": "google.gemma-3-27b-it",
        "output_text": "Hello from ITI Gateway!",
    }
    mock_response.raise_for_status.return_value = None

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        res = client.invoke([
            SystemMessage(content="You are a system prompt."),
            HumanMessage(content="Hello world"),
        ])

        assert res.content == "Hello from ITI Gateway!"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        endpoint = call_kwargs[0][0]
        json_data = call_kwargs[1]["json"]

        assert endpoint == "http://apiaccess.iti.net.eg/api/v1/student/chat"
        assert json_data["model_id"] == "google.gemma-3-27b-it"
        assert json_data["system_prompt"] == "You are a system prompt."
        assert json_data["messages"] == [{"role": "user", "content": "Hello world"}]


def test_chat_iti_structured_output():
    client = ChatITI(
        model_name="google.gemma-3-27b-it",
        api_key="test-key",
        base_url="http://apiaccess.iti.net.eg/api/v1",
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "output_text": '{"summary": "All systems operational", "score": 0.95}',
    }
    mock_response.raise_for_status.return_value = None

    with patch("httpx.Client.post", return_value=mock_response):
        structured_llm = client.with_structured_output(DummySchema)
        result = structured_llm.invoke([
            SystemMessage(content="Analyze system state."),
            HumanMessage(content="Status check"),
        ])

        assert isinstance(result, DummySchema)
        assert result.summary == "All systems operational"
        assert result.score == 0.95


def test_pricing_batch_llm_result_parsing_with_pricing_decisions_alias():
    client = ChatITI(
        model_name="google.gemma-3-27b-it",
        api_key="test-key",
        base_url="http://apiaccess.iti.net.eg/api/v1",
    )

    mock_llm_output = '{"PricingDecisions": [{"product_id": "prod-001", "discount_percentage": 15, "reason": "HIGH risk: Critical expiry and low sales velocity."}]}'

    mock_response = MagicMock()
    mock_response.json.return_value = {"output_text": mock_llm_output}
    mock_response.raise_for_status.return_value = None

    with patch("httpx.Client.post", return_value=mock_response):
        structured_llm = client.with_structured_output(PricingBatchLLMResult)
        result = structured_llm.invoke([
            SystemMessage(content="Recommend pricing."),
            HumanMessage(content="Calculate discount"),
        ])

        assert isinstance(result, PricingBatchLLMResult)
        assert len(result.decisions) == 1
        decision = result.decisions[0]
        assert decision.product_id == "prod-001"
        assert decision.discount_percentage == 15.0
        assert decision.reason == "HIGH risk: Critical expiry and low sales velocity."
        assert decision.confidence == 0.90
