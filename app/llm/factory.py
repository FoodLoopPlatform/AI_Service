from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config.settings import settings


def get_llm(
    model_name: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Constructs and returns the configured ChatOpenAI model instance.

    Configured with model name and API key from settings and temperature=0 by default.
    Instantiated on demand when called. Uses a placeholder API key if none is set
    to allow local structural instantiation without contacting external services.
    """
    resolved_model = model_name or settings.OPENAI_MODEL
    resolved_api_key = api_key if api_key is not None else settings.OPENAI_API_KEY

    # OpenAI SDK requires a non-empty string for client initialization.
    # A placeholder ensures structural instantiation succeeds without real credentials.
    effective_api_key = resolved_api_key if resolved_api_key else "placeholder-api-key"

    return ChatOpenAI(
        model=resolved_model,
        api_key=effective_api_key,
        temperature=temperature,
    )
