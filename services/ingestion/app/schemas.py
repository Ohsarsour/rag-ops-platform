"""Pydantic request/response schemas for the Ingestion Service.

These models declare the HTTP contract for ``POST /ingest``. FastAPI validates
incoming JSON against ``IngestRequest`` automatically and returns HTTP 422 when a
required field is missing or empty.

A deliberate design choice about ``content_type``
-------------------------------------------------
The design says unsupported formats must return HTTP 415, not 422.
If we typed ``content_type`` as a ``Literal["markdown", "text", "pdf"]``, Pydantic
would reject an unknown value with a 422 *before our code ever runs*, so we could
never produce a 415. To keep control, we accept ``content_type`` as a plain
``str`` here and map unsupported values to 415 explicitly in ``main.py``.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """One document submitted for ingestion (design "Ingestion Service").

    Fields:
        source_path: Where the document came from, e.g.
            ``"content/en/docs/concepts/overview.md"``. Stored as Chunk_Metadata
            . ``min_length=1`` makes a missing/empty value a 422.
        source_name: The document's display name, e.g. ``"overview.md"``. Also
            stored as Chunk_Metadata.
        content_type: One of ``"markdown"``, ``"text"``, or ``"pdf"``. Typed as a
            free ``str`` (not a Literal) on purpose so an unsupported value can be
            turned into a 415 rather than Pydantic's 422.
        content: The raw document body. For markdown/text this is the text
            itself; for PDF it is the PDF file bytes encoded as base64 (decoded
            and text-extracted in ``main.py``). ``min_length=1`` means an absent
            body is a 422.
    """

    source_path: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    content_type: str = Field(min_length=1)
    content: str = Field(min_length=1)


class IngestResponse(BaseModel):
    """Success payload returned by ``POST /ingest`` (Req 4.5).

    Fields:
        document_id: Primary key of the newly stored ``documents`` row.
        chunks_persisted: How many chunks were written. This is ``0`` for an
            empty/whitespace-only document, which is still a successful ingest
            (Req 3.6).
    """

    document_id: int
    chunks_persisted: int
