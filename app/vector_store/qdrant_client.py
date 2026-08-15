from qdrant_client import QdrantClient

from app.config.settings import settings


def get_qdrant_client(
    url: str | None = None,
    api_key: str | None = None,
) -> QdrantClient:
    """Constructs and returns a QdrantClient instance using application settings.
    
    Instantiation does not trigger network calls.
    """
    qdrant_url = url or settings.QDRANT_URL
    qdrant_api_key = api_key if api_key is not None else settings.QDRANT_API_KEY
    api_key_val = qdrant_api_key if qdrant_api_key.strip() else None

    return QdrantClient(
        url=qdrant_url,
        api_key=api_key_val,
    )
