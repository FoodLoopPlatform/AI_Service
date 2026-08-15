from unittest.mock import MagicMock, patch
import pytest

from app.config.settings import settings
from app.embeddings import (
    EmbeddingProvider,
    EmbeddingProviderError,
    FakeEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)
from app.embeddings.base import (
    validate_documents_embeddings,
    validate_query_embedding,
)


def test_empty_document_list_behavior():
    """Requirement 7.1: Empty document list returns empty list [] without error."""
    provider = FakeEmbeddingProvider()
    res = provider.embed_documents([])
    assert res == []


def test_single_document_embedding():
    """Requirement 7.2: Single document embedding returns list with 1 vector."""
    provider = FakeEmbeddingProvider(dimension=512)
    res = provider.embed_documents(["Historical pricing snapshot for Organic Milk"])
    assert len(res) == 1
    assert len(res[0]) == 512
    assert isinstance(res[0][0], float)


def test_multiple_documents_embedding():
    """Requirement 7.3: Multiple documents embedding returns list of vectors."""
    provider = FakeEmbeddingProvider(dimension=256)
    docs = ["Doc A", "Doc B", "Doc C"]
    res = provider.embed_documents(docs)
    assert len(res) == 3


def test_number_of_vectors_equals_number_of_documents():
    """Requirement 7.4: Number of returned vectors equals number of input documents."""
    provider = FakeEmbeddingProvider()
    docs = [f"Historical event {i}" for i in range(10)]
    res = provider.embed_documents(docs)
    assert len(res) == len(docs)


def test_consistent_vector_dimensions():
    """Requirement 7.5: All vectors in a batch have consistent dimension."""
    provider = FakeEmbeddingProvider(dimension=128)
    docs = ["Doc 1", "Doc 2", "Doc 3", "Doc 4"]
    res = provider.embed_documents(docs)
    for vec in res:
        assert len(vec) == 128


def test_deterministic_fake_provider_behavior():
    """Requirement 7.6: FakeEmbeddingProvider produces deterministic output across calls."""
    provider = FakeEmbeddingProvider(dimension=64)
    docs = ["Milk event", "Yogurt event"]
    res1 = provider.embed_documents(docs)
    res2 = provider.embed_documents(docs)
    assert res1 == res2


def test_same_text_same_vector():
    """Requirement 7.7: Same text input produces identical vector."""
    provider = FakeEmbeddingProvider(dimension=64)
    v1 = provider.embed_query("Organic MilkCairo")
    v2 = provider.embed_query("Organic MilkCairo")
    assert v1 == v2


def test_different_text_different_vector():
    """Requirement 7.8: Different text produces deterministic but different vector."""
    provider = FakeEmbeddingProvider(dimension=64)
    v1 = provider.embed_query("Product A")
    v2 = provider.embed_query("Product B")
    assert v1 != v2
    assert len(v1) == len(v2) == 64


def test_embed_query_works():
    """Requirement 7.9: embed_query returns a valid single vector."""
    provider = FakeEmbeddingProvider(dimension=128)
    v = provider.embed_query("Find milk near expiry")
    assert isinstance(v, list)
    assert len(v) == 128
    assert all(isinstance(x, float) for x in v)


def test_invalid_provider_output_raises_embedding_provider_error():
    """Requirement 7.10: Invalid provider output (count mismatch, empty vector, dimension mismatch) raises EmbeddingProviderError."""
    # Count mismatch
    with pytest.raises(EmbeddingProviderError) as exc_count:
        validate_documents_embeddings(["doc1", "doc2"], [[0.1, 0.2]])
    assert "Embedding count mismatch" in str(exc_count.value)

    # Empty vector in batch
    with pytest.raises(EmbeddingProviderError) as exc_empty:
        validate_documents_embeddings(["doc1"], [[]])
    assert "Returned empty vector" in str(exc_empty.value)

    # Dimension mismatch
    with pytest.raises(EmbeddingProviderError) as exc_dim:
        validate_documents_embeddings(["doc1", "doc2"], [[0.1, 0.2], [0.1, 0.2, 0.3]])
    assert "Inconsistent vector dimension" in str(exc_dim.value)

    # Empty query vector
    with pytest.raises(EmbeddingProviderError) as exc_query:
        validate_query_embedding("query", [])
    assert "Query embedding vector is empty" in str(exc_query.value)


def test_provider_construction_does_not_make_network_calls():
    """Requirement 7.11: Constructing OpenAIEmbeddingProvider or calling factory does not make network calls."""
    with patch("langchain_openai.OpenAIEmbeddings.embed_documents", side_effect=AssertionError("Network call during init!")):
        provider = OpenAIEmbeddingProvider(api_key="sk-test-key", model="text-embedding-3-small")
        assert provider.model == "text-embedding-3-small"
        assert provider.api_key == "sk-test-key"

        with patch.object(settings, "EMBEDDING_PROVIDER", "openai"):
            factory_provider = get_embedding_provider()
            assert isinstance(factory_provider, OpenAIEmbeddingProvider)


def test_openai_provider_configuration_reads_model():
    """Requirement 7.12: OpenAI provider reads OPENAI_EMBEDDING_MODEL from configuration."""
    provider = OpenAIEmbeddingProvider()
    assert provider.model == settings.OPENAI_EMBEDDING_MODEL
    assert provider.model == "text-embedding-3-small"


def test_missing_api_key_does_not_trigger_network_calls_during_construction():
    """Requirement 7.13: Missing API key constructs without error, but raises EmbeddingProviderError when invoked."""
    provider = OpenAIEmbeddingProvider(api_key="")
    assert provider.api_key == ""

    with pytest.raises(EmbeddingProviderError) as exc_info:
        provider.embed_documents(["some doc"])
    assert "API key is missing or empty" in str(exc_info.value)

    with pytest.raises(EmbeddingProviderError) as exc_query_info:
        provider.embed_query("some query")
    assert "API key is missing or empty" in str(exc_query_info.value)


def test_provider_failures_propagate_as_embedding_provider_error():
    """Requirement 7.14: External provider exceptions propagate as EmbeddingProviderError."""
    provider = OpenAIEmbeddingProvider(api_key="sk-fake-key")

    mock_client = MagicMock()
    mock_client.embed_documents.side_effect = RuntimeError("Connection timeout to OpenAI embeddings")
    mock_client.embed_query.side_effect = RuntimeError("OpenAI rate limit exceeded")

    with patch.object(provider, "_get_client", return_value=mock_client):
        with pytest.raises(EmbeddingProviderError) as exc_docs:
            provider.embed_documents(["doc1"])
        assert "OpenAI embed_documents failed" in str(exc_docs.value)
        assert "Connection timeout" in str(exc_docs.value)

        with pytest.raises(EmbeddingProviderError) as exc_query:
            provider.embed_query("query")
        assert "OpenAI embed_query failed" in str(exc_query.value)
        assert "rate limit" in str(exc_query.value)
