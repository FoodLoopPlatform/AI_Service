import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config.settings import settings


def get_llm(
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Constructs and returns the configured ChatOpenAI model instance.

    Configured with model name, API key, and optional base_url from settings.
    Instantiated on demand when called. Uses a placeholder API key if none is set
    to allow local structural instantiation without contacting external services.
    """
    resolved_model = model_name or settings.OPENAI_MODEL
    resolved_api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
    resolved_base_url = base_url if base_url is not None else settings.OPENAI_BASE_URL

    # Detect ITI Gateway endpoint and return custom ChatITI adapter
    if resolved_base_url and "iti.net.eg" in resolved_base_url:
        from app.llm.iti import ChatITI
        return ChatITI(
            model_name=resolved_model,
            api_key=resolved_api_key or "",
            base_url=resolved_base_url,
            temperature=temperature,
            timeout=settings.OPENAI_TIMEOUT_SECONDS,
        )

    # OpenAI SDK requires a non-empty string for client initialization.
    # A placeholder ensures structural instantiation succeeds without real credentials.
    effective_api_key = resolved_api_key if resolved_api_key else "placeholder-api-key"

    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "api_key": effective_api_key,
        "temperature": temperature,
        "timeout": settings.OPENAI_TIMEOUT_SECONDS,
        "max_retries": 2,
    }

    if resolved_base_url and resolved_base_url.strip():
        kwargs["base_url"] = resolved_base_url.strip()

    return ChatOpenAI(**kwargs)


def verify_openai_connection(
    llm: BaseChatModel | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Performs an explicit opt-in real API call to verify the configured OpenAI chat model.

    Uses a minimal, harmless prompt ('What is 2 + 2? Return the number only.')
    Returns status and latency without revealing credentials or business data.
    """
    start_time = time.perf_counter()
    target_llm = llm or get_llm(model_name=model_name, api_key=api_key, base_url=base_url)

    response = target_llm.invoke("What is 2 + 2? Return the number only.")
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    content = str(response.content).strip()
    return {
        "status": "ok",
        "latency_ms": duration_ms,
        "model": getattr(target_llm, "model_name", settings.OPENAI_MODEL),
        "response_sample": content,
    }
