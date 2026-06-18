"""Endpoint tests for the Embedding Service (task 2.2).

The real `all-MiniLM-L6-v2` model is heavy and would require a network
download, so these tests stub the shared model instead. A small deterministic
`FakeModel` is installed onto the module-level `model_holder`, which lets the
endpoints exercise the real `_encode_texts` shared path, response schemas, and
readiness logic without the actual sentence-transformers model (Req 9.5).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import EMBEDDING_DIMENSIONS


class FakeModel:
    """Deterministic stand-in for SentenceTransformer.

    `encode` mirrors the real signature (`normalize_embeddings` keyword) and
    returns a numpy ndarray of shape (N, 384), so the endpoints' numpy->list
    conversion is exercised. Each vector is a deterministic function of the
    text, so identical text always yields identical vectors (lets us assert
    determinism and single/batch equivalence).
    """

    def encode(self, texts, normalize_embeddings=False):
        # The endpoints always pass a list of texts through the shared helper.
        vectors = [self._vector_for(text) for text in texts]
        return np.array(vectors, dtype=np.float32)

    @staticmethod
    def _vector_for(text: str) -> list:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(EMBEDDING_DIMENSIONS).tolist()


@pytest.fixture
def client(monkeypatch):
    """TestClient backed by the FakeModel.

    The fake is installed directly on the holder so the lifespan handler's
    idempotent `load()` becomes a no-op (it returns early when a model is
    already present) and never imports sentence-transformers.
    """
    fake = FakeModel()
    monkeypatch.setattr(main.model_holder, "_model", fake)
    with TestClient(main.app) as test_client:
        yield test_client


# --- /embed ---------------------------------------------------------------


def test_embed_returns_384_dim_finite_floats(client):
    resp = client.post("/embed", json={"text": "hello world"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dimensions"] == 384
    assert len(body["embedding"]) == 384
    assert all(isinstance(v, float) for v in body["embedding"])
    assert all(np.isfinite(v) for v in body["embedding"])


def test_embed_whitespace_only_text_is_embedded_normally(client):
    # Whitespace-only text is valid and produces a full 384-dim vector (Req 1.4).
    resp = client.post("/embed", json={"text": "   "})
    assert resp.status_code == 200
    assert len(resp.json()["embedding"]) == 384


def test_embed_is_deterministic_for_same_text(client):
    first = client.post("/embed", json={"text": "k8s"}).json()["embedding"]
    second = client.post("/embed", json={"text": "k8s"}).json()["embedding"]
    assert first == second


def test_embed_rejects_empty_text_with_422(client):
    resp = client.post("/embed", json={"text": ""})
    assert resp.status_code == 422


# --- /embed/batch ---------------------------------------------------------


def test_batch_returns_one_vector_per_input(client):
    texts = ["alpha", "beta", "gamma"]
    resp = client.post("/embed/batch", json={"texts": texts})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dimensions"] == 384
    assert len(body["embeddings"]) == len(texts)
    for vector in body["embeddings"]:
        assert len(vector) == 384
        assert all(np.isfinite(v) for v in vector)


def test_batch_is_positionally_aligned_with_single(client):
    # Each batch vector must equal the single-embed vector for that same text
    # (Req 2.2 alignment + Req 2.3 single/batch equivalence via shared path).
    texts = ["one", "two", "three"]
    batch = client.post("/embed/batch", json={"texts": texts}).json()["embeddings"]
    for i, text in enumerate(texts):
        single = client.post("/embed", json={"text": text}).json()["embedding"]
        assert batch[i] == single


def test_batch_rejects_empty_list_with_422(client):
    resp = client.post("/embed/batch", json={"texts": []})
    assert resp.status_code == 422


# --- /healthz -------------------------------------------------------------


def test_healthz_ok_when_model_loaded(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_healthz_503_when_model_not_loaded(monkeypatch):
    # Without a loaded model, readiness must report 503 (not 200) so compose
    # healthchecks do not route traffic before the model is usable.
    monkeypatch.setattr(main.model_holder, "_model", None)
    # Build the client without triggering the lifespan loader (which would
    # attempt to import the real model); call the route function directly.
    response = main.healthz()
    assert response.status_code == 503
