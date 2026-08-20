from unittest.mock import patch
from langchain_openai import ChatOpenAI

from app.config.settings import settings
from app.llm import get_llm
from app.llm.iti import ChatITI


def test_settings_read_openai_model():
    assert settings.OPENAI_MODEL is not None
    assert isinstance(settings.OPENAI_MODEL, str)


def test_get_llm_uses_configured_model():
    llm = get_llm()
    assert llm.model_name == settings.OPENAI_MODEL


def test_get_llm_returns_chat_iti_for_iti_base_url():
    llm = get_llm(base_url="http://apiaccess.iti.net.eg/api/v1")
    assert isinstance(llm, ChatITI)
    assert llm.model_name == settings.OPENAI_MODEL


def test_get_llm_returns_chat_openai_for_standard_url():
    llm = get_llm(base_url="https://api.openai.com/v1")
    assert isinstance(llm, ChatOpenAI)


def test_get_llm_temperature_is_zero():
    llm = get_llm()
    assert llm.temperature == 0.0


def test_factory_does_not_make_api_request_on_construction():
    with patch("httpx.Client.send") as mock_send, patch(
        "httpx.AsyncClient.send"
    ) as mock_async_send:
        llm = get_llm()
        assert llm is not None
        mock_send.assert_not_called()
        mock_async_send.assert_not_called()
