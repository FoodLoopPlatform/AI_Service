import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.config.settings import settings
from app.embeddings import (
    EmbeddingProviderError,
    FakeEmbeddingProvider,
    LocalBGEEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)
from app.agents.pricing.retriever import VectorPricingKnowledgeRetriever
from app.schemas.pricing_knowledge import PricingKnowledgeItem
from app.schemas.pricing_knowledge_document import PricingKnowledgeDocument
from app.vector_store.in_memory import InMemoryVectorStore


def test_1_bge_m3_provider_construction():
    provider = LocalBGEEmbeddingProvider(model_name="BAAI/bge-m3", device="cpu")
    assert provider.model_name == "BAAI/bge-m3"
    assert provider.device == "cpu"
    assert provider.expected_dimension == 1024
    assert provider._model is None  # Lazy loading invariant


def test_2_bge_m3_lazy_model_loading():
    provider = LocalBGEEmbeddingProvider(model_name="BAAI/bge-m3", device="cpu")
    assert provider._model is None

    mock_st_class = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_instance.encode.return_value = np.zeros((1, 1024), dtype=np.float32)
    mock_st_class.return_value = mock_model_instance

    mock_st_module = MagicMock()
    mock_st_module.SentenceTransformer = mock_st_class

    with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
        vecs = provider.embed_documents(["Test document"])
        assert mock_st_class.call_count == 1
        assert provider._model is mock_model_instance
        assert len(vecs[0]) == 1024

        # Second call reuses loaded instance
        provider.embed_documents(["Another document"])
        assert mock_st_class.call_count == 1  # Not called again


def test_3_bge_m3_device_validation_cuda_unavailable():
    provider = LocalBGEEmbeddingProvider(device="cuda")
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": mock_torch}):
        with pytest.raises(EmbeddingProviderError, match="CUDA device requested"):
            provider.embed_documents(["Test"])


def test_4_bge_m3_batch_documents_mocked():
    mock_st_instance = MagicMock()
    mock_st_instance.encode.return_value = np.random.rand(3, 1024).astype(np.float32)

    provider = LocalBGEEmbeddingProvider()
    provider._model = mock_st_instance

    docs = [
        "High inventory pressure and short remaining shelf life.",
        "المخزون مرتفع والمنتج قريب من انتهاء الصلاحية.",
        "High inventory pressure مع قرب انتهاء الصلاحية.",
    ]

    embeddings = provider.embed_documents(docs)
    assert len(embeddings) == 3
    for vec in embeddings:
        assert len(vec) == 1024


def test_5_bge_m3_batch_queries_mocked():
    mock_st_instance = MagicMock()
    mock_st_instance.encode.return_value = np.random.rand(2, 1024).astype(np.float32)

    provider = LocalBGEEmbeddingProvider()
    provider._model = mock_st_instance

    queries = ["What is the expiry risk?", "ما هو خطر انتهاء الصلاحية؟"]
    embeddings = provider.embed_queries(queries)

    assert len(embeddings) == 2
    for vec in embeddings:
        assert len(vec) == 1024


def test_6_bge_m3_single_query_mocked():
    mock_st_instance = MagicMock()
    mock_st_instance.encode.return_value = np.random.rand(1024).astype(np.float32)

    provider = LocalBGEEmbeddingProvider()
    provider._model = mock_st_instance

    vec = provider.embed_query("Low demand and high price")
    assert len(vec) == 1024


def test_7_factory_provider_selection_bge_m3():
    with patch.object(settings, "EMBEDDING_PROVIDER", "local_bge_m3"):
        provider = get_embedding_provider()
        assert isinstance(provider, LocalBGEEmbeddingProvider)


def test_8_fake_and_openai_providers_remain_functional():
    fake_provider = FakeEmbeddingProvider(dimension=1024)
    assert len(fake_provider.embed_query("test")) == 1024

    openai_provider = OpenAIEmbeddingProvider(model="text-embedding-3-small", api_key="sk-fake")
    assert openai_provider.model == "text-embedding-3-small"


def test_9_no_openai_calls_when_bge_m3_active():
    with patch.object(settings, "EMBEDDING_PROVIDER", "local_bge_m3"):
        with patch("app.embeddings.openai.OpenAIEmbeddingProvider.embed_query") as mock_openai:
            provider = get_embedding_provider()
            assert isinstance(provider, LocalBGEEmbeddingProvider)
            mock_openai.assert_not_called()


@pytest.mark.skipif(
    os.getenv("RUN_EXTERNAL_INTEGRATION_TESTS") != "true"
    and os.getenv("RUN_BGE_M3_TESTS") != "true",
    reason="Opt-in BGE-M3 model test. Set RUN_BGE_M3_TESTS=true to download/run local model inference.",
)
def test_10_live_bge_m3_multilingual_arabic_english_crosslingual_retrieval():
    """Live opt-in integration test executing BGE-M3 local model inference for Arabic & English cross-lingual retrieval."""
    provider = LocalBGEEmbeddingProvider(model_name="BAAI/bge-m3", device="cpu")
    store = InMemoryVectorStore()

    # 1. Embed and store Arabic historical knowledge document
    arabic_content = "المنتج قريب من انتهاء الصلاحية والمخزون مرتفع."
    doc = PricingKnowledgeDocument(
        document_id="doc-arabic-001",
        store_id="store-100",
        product_id="prod-dairy-1",
        content=arabic_content,
        metadata={"category": "Dairy", "store_id": "store-100", "product_id": "prod-dairy-1"},
    )

    doc_embeddings = provider.embed_documents([arabic_content])
    assert len(doc_embeddings) == 1
    assert len(doc_embeddings[0]) == 1024

    store.upsert([doc], doc_embeddings)

    # 2. Query in English
    english_query = "The product is near expiry and inventory is high."
    query_vec = provider.embed_query(english_query)
    assert len(query_vec) == 1024

    # 3. Perform vector search
    results = store.search(
        query_vec,
        store_id="store-100",
        product_id="prod-dairy-1",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].content == arabic_content
    # Cross-lingual cosine similarity score should be high (> 0.5)
    assert results[0].relevance_score > 0.5
