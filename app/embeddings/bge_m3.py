import os

if "HF_HOME" not in os.environ and os.path.exists("/media/dell/Data3"):
    os.environ["HF_HOME"] = "/media/dell/Data3/.cache/huggingface"

import logging
from typing import Any

from app.config.settings import settings
from app.embeddings.base import (
    EmbeddingProvider,
    EmbeddingProviderError,
    validate_documents_embeddings,
    validate_query_embedding,
)

logger = logging.getLogger(__name__)




class LocalBGEEmbeddingProvider(EmbeddingProvider):
    """Local BAAI/bge-m3 multilingual embedding provider via sentence-transformers.
    
    Supports 100+ languages (including Arabic and English), 1024-dimensional dense vectors,
    lazy model loading, and configurable CPU/CUDA device execution.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        expected_dimension: int | None = None,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = (device or settings.EMBEDDING_DEVICE).lower().strip()
        self.expected_dimension = expected_dimension or settings.EMBEDDING_VECTOR_SIZE
        self._model: Any = None

    def _get_model(self) -> Any:
        """Lazy-initializes and caches the SentenceTransformer model instance on first use."""
        if self._model is not None:
            return self._model

        # 1. Device validation
        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    raise EmbeddingProviderError(
                        "CUDA device requested for BGE-M3 embedding provider, "
                        "but PyTorch CUDA is not available on this system."
                    )
            except ImportError as e:
                raise EmbeddingProviderError(
                    "CUDA device requested, but PyTorch could not be imported."
                ) from e
        elif self.device != "cpu":
            raise EmbeddingProviderError(
                f"Unsupported EMBEDDING_DEVICE setting: '{self.device}'. Must be 'cpu' or 'cuda'."
            )

        # 2. Lazy model loading
        logger.info(
            "Lazy-loading local multilingual embedding model '%s' on device '%s'...",
            self.model_name,
            self.device,
        )

        try:
            import os
            if "HF_HOME" not in os.environ and os.path.exists("/media/dell/Data3"):
                os.environ["HF_HOME"] = "/media/dell/Data3/.cache/huggingface"

            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Successfully loaded embedding model '%s'.", self.model_name)
            return self._model

        except Exception as e:
            err_msg = f"Failed to load local BGE-M3 embedding model '{self.model_name}': {e}"
            logger.error(err_msg)
            raise EmbeddingProviderError(err_msg) from e

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Generate 1024-d normalized vector embeddings for a list of document strings."""
        if not documents:
            return []

        try:
            model = self._get_model()
            embeddings_ndarray = model.encode(
                documents,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embeddings: list[list[float]] = [vec.tolist() for vec in embeddings_ndarray]
        except EmbeddingProviderError:
            raise
        except Exception as e:
            raise EmbeddingProviderError(f"Error generating document embeddings with BGE-M3: {e}") from e

        # Validate vector dimensions and non-emptiness
        validate_documents_embeddings(documents, embeddings)
        if embeddings and len(embeddings[0]) != self.expected_dimension:
            raise EmbeddingProviderError(
                f"Vector dimension mismatch: expected {self.expected_dimension}, got {len(embeddings[0])}."
            )

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Generate a 1024-d normalized vector embedding for a single query string."""
        if not query or not isinstance(query, str):
            raise EmbeddingProviderError("Query string must be a non-empty string.")

        try:
            model = self._get_model()
            embedding_ndarray = model.encode(
                query,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embedding: list[float] = embedding_ndarray.tolist()
        except EmbeddingProviderError:
            raise
        except Exception as e:
            raise EmbeddingProviderError(f"Error generating query embedding with BGE-M3: {e}") from e

        validate_query_embedding(query, embedding)
        if len(embedding) != self.expected_dimension:
            raise EmbeddingProviderError(
                f"Query vector dimension mismatch: expected {self.expected_dimension}, got {len(embedding)}."
            )

        return embedding

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Generate 1024-d normalized vector embeddings for a list of query strings in batch."""
        if not queries:
            return []

        try:
            model = self._get_model()
            embeddings_ndarray = model.encode(
                queries,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embeddings: list[list[float]] = [vec.tolist() for vec in embeddings_ndarray]
        except EmbeddingProviderError:
            raise
        except Exception as e:
            raise EmbeddingProviderError(f"Error generating query batch embeddings with BGE-M3: {e}") from e

        validate_documents_embeddings(queries, embeddings)
        if embeddings and len(embeddings[0]) != self.expected_dimension:
            raise EmbeddingProviderError(
                f"Vector dimension mismatch: expected {self.expected_dimension}, got {len(embeddings[0])}."
            )

        return embeddings
