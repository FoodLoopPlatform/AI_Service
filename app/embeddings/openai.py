from app.config.settings import settings
from app.embeddings.base import (
    EmbeddingProvider,
    EmbeddingProviderError,
    validate_documents_embeddings,
    validate_query_embedding,
)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI implementation of EmbeddingProvider using LangChain OpenAIEmbeddings."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.model = settings.OPENAI_EMBEDDING_MODEL if model is None else model
        self.api_key = settings.OPENAI_API_KEY if api_key is None else api_key

        self._client = None

    def _get_client(self):
        if not self.api_key or not self.api_key.strip():
            raise EmbeddingProviderError("OpenAI API key is missing or empty.")

        if self._client is None:
            try:
                from langchain_openai import OpenAIEmbeddings

                self._client = OpenAIEmbeddings(
                    model=self.model,
                    openai_api_key=self.api_key,
                )
            except Exception as e:
                raise EmbeddingProviderError(
                    f"Failed to initialize OpenAIEmbeddings client: {e}"
                ) from e
        return self._client

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a batch of document strings using OpenAI API."""
        if not documents:
            return []

        client = self._get_client()
        try:
            embeddings = client.embed_documents(documents)
        except Exception as e:
            raise EmbeddingProviderError(f"OpenAI embed_documents failed: {e}") from e

        validate_documents_embeddings(documents, embeddings)
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Generate a vector embedding for a query string using OpenAI API."""
        client = self._get_client()
        try:
            embedding = client.embed_query(query)
        except Exception as e:
            raise EmbeddingProviderError(f"OpenAI embed_query failed: {e}") from e

        validate_query_embedding(query, embedding)
        return embedding

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a batch of query strings using OpenAI API."""
        if not queries:
            return []

        client = self._get_client()
        try:
            embeddings = client.embed_documents(queries)
        except Exception as e:
            raise EmbeddingProviderError(f"OpenAI embed_queries failed: {e}") from e

        validate_documents_embeddings(queries, embeddings)
        return embeddings

