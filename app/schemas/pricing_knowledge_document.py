from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class PricingKnowledgeDocument(BaseModel):
    """Retrieval-ready document model containing natural language historical facts and metadata."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(..., min_length=1, description="Unique document identifier")
    store_id: str = Field(..., min_length=1, description="Originating store ID")
    product_id: str = Field(..., min_length=1, description="Originating product ID")
    content: str = Field(..., min_length=1, description="Natural language description of historical event")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured metadata suitable for vector database filtering",
    )
