import logging
import time
from typing import Callable

from app.agents.pricing.knowledge_builder import build_pricing_knowledge_document
from app.embeddings import EmbeddingProvider, EmbeddingProviderError, get_embedding_provider
from app.schemas.historical_pricing import HistoricalPricingEvent
from app.schemas.pricing_knowledge_document import PricingKnowledgeDocument
from app.schemas.pricing_knowledge_ingestion import (
    HistoricalPricingIngestionRequest,
    HistoricalPricingIngestionResponse,
)
from app.vector_store import VectorStore, VectorStoreError, get_vector_store

logger = logging.getLogger(__name__)


class HistoricalPricingIngestionService:
    """Service layer executing authoritative historical pricing event ingestion,
    document transformation, batch embedding, and idempotent vector database upsert.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        knowledge_builder: Callable[[HistoricalPricingEvent], PricingKnowledgeDocument] | None = None,
    ):
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._knowledge_builder = knowledge_builder or build_pricing_knowledge_document

    def _get_embedding_provider(self) -> EmbeddingProvider:
        if self._embedding_provider is not None:
            return self._embedding_provider
        return get_embedding_provider()

    def _get_vector_store(self) -> VectorStore:
        if self._vector_store is not None:
            return self._vector_store
        return get_vector_store()

    def ingest(
        self,
        request: HistoricalPricingIngestionRequest | list[HistoricalPricingEvent],
        request_id: str | None = None,
    ) -> HistoricalPricingIngestionResponse:
        """Ingests a batch of HistoricalPricingEvent snapshots into the vector database.
        
        This operation is atomic and idempotent by event_id. Re-ingesting an event_id
        re-embeds and upserts the stored record without creating duplicate entries.
        """
        start_time = time.perf_counter()

        if isinstance(request, HistoricalPricingIngestionRequest):
            events = request.events
        else:
            events = request

        req_id_str = f" [request_id={request_id}]" if request_id else ""
        logger.info("Starting historical pricing ingestion batch of %d events%s", len(events), req_id_str)

        if not events:
            return HistoricalPricingIngestionResponse(
                accepted_count=0,
                upserted_count=0,
                failed_count=0,
                document_ids=[],
            )

        # 1. Transform events to deterministic knowledge documents
        documents: list[PricingKnowledgeDocument] = []
        document_ids: list[str] = []
        for event in events:
            doc = self._knowledge_builder(event)
            documents.append(doc)
            document_ids.append(doc.document_id)

        contents = [doc.content for doc in documents]

        # 2. Batch generate embeddings for all documents
        emb_start = time.perf_counter()
        provider = self._get_embedding_provider()
        try:
            embeddings = provider.embed_documents(contents)
        except Exception as exc:
            logger.error(
                "Batch embedding generation failed during historical ingestion%s: %s",
                req_id_str,
                exc,
            )
            if isinstance(exc, EmbeddingProviderError):
                raise
            raise EmbeddingProviderError(f"Batch embedding generation failed: {exc}") from exc

        emb_duration_ms = (time.perf_counter() - emb_start) * 1000.0

        if len(embeddings) != len(documents):
            err_msg = (
                f"Embedding count mismatch during ingestion: generated {len(embeddings)} "
                f"embeddings for {len(documents)} documents."
            )
            logger.error(err_msg + req_id_str)
            raise EmbeddingProviderError(err_msg)

        # 3. Upsert documents and embeddings into vector store
        store_start = time.perf_counter()
        store = self._get_vector_store()
        try:
            store.upsert(documents, embeddings)
        except Exception as exc:
            logger.error(
                "Vector store upsert failed during historical ingestion%s: %s",
                req_id_str,
                exc,
            )
            if isinstance(exc, VectorStoreError):
                raise
            raise VectorStoreError(f"Vector store upsert failed: {exc}") from exc

        store_duration_ms = (time.perf_counter() - store_start) * 1000.0
        total_duration_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            "Successfully ingested %d historical pricing documents%s (emb: %.2fms, store: %.2fms, total: %.2fms)",
            len(documents),
            req_id_str,
            emb_duration_ms,
            store_duration_ms,
            total_duration_ms,
        )

        return HistoricalPricingIngestionResponse(
            accepted_count=len(events),
            upserted_count=len(documents),
            failed_count=0,
            document_ids=document_ids,
        )


def get_ingestion_service() -> HistoricalPricingIngestionService:
    """Factory helper returning a HistoricalPricingIngestionService initialized with active providers."""
    return HistoricalPricingIngestionService()
