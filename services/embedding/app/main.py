"""Embedding Service FastAPI application.

This module wires the model loader into the application lifecycle and exposes
the three HTTP endpoints (task 2.2). The model is loaded exactly once in the
lifespan handler at startup and released at shutdown; it is also exposed on
`app.state` so endpoints reuse the single shared instance (Req 1.6).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import List

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .model import model_holder
from .schemas import (
    BatchEmbedRequest,
    BatchEmbedResponse,
    EmbedRequest,
    EmbedResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the embedding model once on startup; release it on shutdown.

    Loading here (rather than per request) means the model load cost is paid
    a single time and the same instance serves every request (Req 1.6).
    """
    model_holder.load()
    app.state.model_holder = model_holder
    try:
        yield
    finally:
        model_holder.unload()


app = FastAPI(title="Embedding Service", lifespan=lifespan)


def _encode_texts(texts: List[str]) -> List[List[float]]:
    """Encode a list of texts into a list of plain-float vectors.

    This is the single shared encode path used by BOTH `/embed` and
    `/embed/batch`. Routing every request through the same
    `model.encode(..., normalize_embeddings=False)` call guarantees that a
    text embedded singly and the same text embedded in a batch produce an
    identical vector (Req 1.5, 2.3 / design Property 2).

    The model's numpy output is converted to plain Python ``float`` lists so
    the response is JSON-serialisable and each element is a finite float
    (Req 1.2).
    """
    model = model_holder.model
    # normalize_embeddings=False keeps the raw, deterministic model output and
    # must match across the single and batch paths.
    vectors = model.encode(texts, normalize_embeddings=False)
    # SentenceTransformer returns a numpy ndarray; .tolist() yields nested
    # Python floats. Fall back gracefully if a plain list is returned.
    to_list = getattr(vectors, "tolist", None)
    rows = to_list() if callable(to_list) else vectors
    return [[float(value) for value in row] for row in rows]


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    """Embed a single text into one 384-dim vector (Req 1.1, 1.2, 1.4)."""
    embedding = _encode_texts([request.text])[0]
    return EmbedResponse(embedding=embedding)


@app.post("/embed/batch", response_model=BatchEmbedResponse)
def embed_batch(request: BatchEmbedRequest) -> BatchEmbedResponse:
    """Embed a list of texts into positionally-aligned vectors (Req 2.1, 2.2)."""
    embeddings = _encode_texts(request.texts)
    return BatchEmbedResponse(embeddings=embeddings)


@app.get("/healthz")
def healthz() -> JSONResponse:
    """Readiness probe: 200 only once the model is loaded.

    Returns 503 until `model_holder.is_ready()` is True so compose
    healthchecks gate dependent services on a usable model (design "Common
    conventions").
    """
    if model_holder.is_ready():
        return JSONResponse(status_code=200, content={"status": "ok"})
    return JSONResponse(status_code=503, content={"status": "loading"})
