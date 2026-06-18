"""Pydantic request/response schemas for the Embedding Service.

These models declare the input contract so that FastAPI returns HTTP 422
automatically for violations (length/size constraints), satisfying the
validation requirements without custom code. See design "Embedding Service".
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .config import EMBEDDING_DIMENSIONS

# Per-element constraints for batch embedding (Req 2.5).
MAX_BATCH_TEXT_LENGTH = 8192


class EmbedRequest(BaseModel):
    """Single-text embedding request .

    `min_length=1` makes a missing or empty `text` field a 422 automatically
    . Whitespace-only text has length >= 1, so it passes validation
    and is embedded normally.
    """

    text: str = Field(min_length=1, max_length=10_000)


class EmbedResponse(BaseModel):
    """Single-text embedding response"""

    embedding: list[float]  # length 384, finite floats
    dimensions: int = EMBEDDING_DIMENSIONS


class BatchEmbedRequest(BaseModel):
    """Batch embedding request (Req 2).

    `min_length=1` rejects an empty list ; `max_length=1000` rejects
    batches larger than 1000 . The per-element field validator
    enforces that each element is a non-empty string of at most 8192
    characters and reports the index and reason of the first invalid element
    .
    """

    texts: list[str] = Field(min_length=1, max_length=1000)

    @field_validator("texts")
    @classmethod
    def _validate_elements(cls, texts: list[str]) -> list[str]:
        for index, element in enumerate(texts):
            if not isinstance(element, str):
                raise ValueError(
                    f"element at index {index} is invalid: must be a string"
                )
            if len(element) == 0:
                raise ValueError(
                    f"element at index {index} is invalid: must be non-empty"
                )
            if len(element) > MAX_BATCH_TEXT_LENGTH:
                raise ValueError(
                    f"element at index {index} is invalid: exceeds maximum "
                    f"length of {MAX_BATCH_TEXT_LENGTH} characters"
                )
        return texts


class BatchEmbedResponse(BaseModel):
    """Batch embedding response (Req 2.1, 2.2)."""

    embeddings: list[list[float]]  # N vectors, aligned to input order
    dimensions: int = EMBEDDING_DIMENSIONS
