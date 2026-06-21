"""Embedding model loader and holder.

The Embedding Service is the only component that owns the embedding model.
It loads `all-MiniLM-L6-v2` exactly once at startup (via the FastAPI lifespan
handler in `main.py`) and reuses the same instance for every request, so the
load cost is paid once rather than per request (design "Embedding Service").
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .config import MODEL_NAME

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from sentence_transformers import SentenceTransformer


class ModelHolder:
    """Holds the single shared embedding model instance.

    A module-level holder keeps the loaded model accessible to endpoints
    while making readiness explicit: `is_ready()` is True only after the
    model has finished loading, which `/healthz` relies on.
    """

    def __init__(self) -> None:
        self._model: Optional["SentenceTransformer"] = None

    def load(self) -> None:
        """Load the embedding model once. Idempotent."""
        if self._model is not None:
            return
        # Imported lazily so the module can be imported without the heavy
        # dependency present (e.g. for schema-only unit tests).
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(MODEL_NAME)

    def unload(self) -> None:
        """Release the model reference (used on shutdown)."""
        self._model = None

    def is_ready(self) -> bool:
        """Return True once the model has been loaded."""
        return self._model is not None

    @property
    def model(self) -> "SentenceTransformer":
        """Return the loaded model, or raise if it is not ready."""
        if self._model is None:
            raise RuntimeError("Embedding model has not been loaded yet")
        return self._model


# Single shared holder reused across the application.
model_holder = ModelHolder()
