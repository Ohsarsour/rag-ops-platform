"""Unit tests for the Embedding Service Pydantic schemas (task 2.1).

These cover the declarative validation contract: length/size constraints on
the single and batch requests, and the per-element batch field validator that
reports the index and reason of the first invalid element.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    BatchEmbedRequest,
    BatchEmbedResponse,
    EmbedRequest,
    EmbedResponse,
)


# --- EmbedRequest ---------------------------------------------------------


def test_embed_request_accepts_normal_text():
    req = EmbedRequest(text="hello world")
    assert req.text == "hello world"


def test_embed_request_accepts_whitespace_only_text():
    # Whitespace-only text has length >= 1 and must pass validation (Req 1.4).
    req = EmbedRequest(text="   ")
    assert req.text == "   "


def test_embed_request_rejects_empty_text():
    # Missing/empty text -> 422 via min_length (Req 1.3).
    with pytest.raises(ValidationError):
        EmbedRequest(text="")


def test_embed_request_accepts_max_length_text():
    req = EmbedRequest(text="a" * 10_000)
    assert len(req.text) == 10_000


def test_embed_request_rejects_text_over_max_length():
    # text > 10,000 chars -> 422 (Req 1.7).
    with pytest.raises(ValidationError):
        EmbedRequest(text="a" * 10_001)


# --- BatchEmbedRequest ----------------------------------------------------


def test_batch_request_accepts_valid_list():
    req = BatchEmbedRequest(texts=["a", "b", "c"])
    assert req.texts == ["a", "b", "c"]


def test_batch_request_rejects_empty_list():
    # Empty list -> 422 (Req 2.4).
    with pytest.raises(ValidationError):
        BatchEmbedRequest(texts=[])


def test_batch_request_accepts_max_batch_size():
    req = BatchEmbedRequest(texts=["x"] * 1000)
    assert len(req.texts) == 1000


def test_batch_request_rejects_batch_over_max_size():
    # More than 1000 items -> 422 (Req 2.6).
    with pytest.raises(ValidationError):
        BatchEmbedRequest(texts=["x"] * 1001)


def test_batch_request_rejects_empty_element_with_index_and_reason():
    # Invalid element -> 422 identifying which element and why (Req 2.5).
    with pytest.raises(ValidationError) as exc_info:
        BatchEmbedRequest(texts=["ok", "", "also ok"])
    message = str(exc_info.value)
    assert "index 1" in message
    assert "non-empty" in message


def test_batch_request_rejects_oversized_element_with_index_and_reason():
    with pytest.raises(ValidationError) as exc_info:
        BatchEmbedRequest(texts=["ok", "a" * 8193])
    message = str(exc_info.value)
    assert "index 1" in message
    assert "8192" in message


def test_batch_request_accepts_element_at_max_length():
    req = BatchEmbedRequest(texts=["a" * 8192])
    assert len(req.texts[0]) == 8192


def test_batch_request_reports_first_invalid_element():
    # When multiple elements are invalid, the first one is reported.
    with pytest.raises(ValidationError) as exc_info:
        BatchEmbedRequest(texts=["ok", "", "a" * 8193])
    assert "index 1" in str(exc_info.value)


# --- Responses ------------------------------------------------------------


def test_embed_response_defaults_dimensions_to_384():
    resp = EmbedResponse(embedding=[0.0] * 384)
    assert resp.dimensions == 384


def test_batch_embed_response_defaults_dimensions_to_384():
    resp = BatchEmbedResponse(embeddings=[[0.0] * 384, [0.1] * 384])
    assert resp.dimensions == 384
