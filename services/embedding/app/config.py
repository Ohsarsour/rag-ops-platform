"""Configuration for the Embedding Service.

All settings are read from environment variables so nothing is hardcoded
(see design "Common conventions"). The model name is configurable so the
embedding model can be swapped without code changes.
"""
from __future__ import annotations

import os

# The sentence-transformers model used to produce every embedding (Req 1.6).
# Configurable via env var; defaults to the Phase 1 choice.
MODEL_NAME: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Expected embedding dimensionality for all-MiniLM-L6-v2 (Req 1.1).
EMBEDDING_DIMENSIONS: int = 384
