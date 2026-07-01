"""Smoke check for the Option B chunker: confirm every chunk actually fits
under all-MiniLM-L6-v2's 256-token input limit (content + 2 special markers),
plus the usual bounds/reconstruction invariants. Throwaway diagnostic."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "ingestion"))

from app.chunker import (  # noqa: E402
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    OVERLAP_MAX_TOKENS,
    OVERLAP_MIN_TOKENS,
    chunk_text,
    reconstruct,
    _get_tokenizer,
)

PARAGRAPH = (
    "A Pod is the smallest deployable unit of computing that you can create and "
    "manage in Kubernetes. A Pod is a group of one or more containers, with shared "
    "storage and network resources, and a specification for how to run the "
    "containers. A Pod's contents are always co-located and co-scheduled, and run "
    "in a shared context. A Pod models an application-specific logical host. "
)
doc = PARAGRAPH * 12  # well over one chunk

chunks = chunk_text(doc)
tok = _get_tokenizer()

print(f"bounds: MIN={MIN_CHUNK_TOKENS} MAX={MAX_CHUNK_TOKENS} "
      f"overlap={OVERLAP_MIN_TOKENS}..{OVERLAP_MAX_TOKENS}")
print(f"produced {len(chunks)} chunks from {len(doc)} chars\n")

worst_model_tokens = 0
for c in chunks:
    # What the embedding model will actually see, including [CLS] + [SEP].
    model_tokens = len(tok.encode(c.content, add_special_tokens=True))
    worst_model_tokens = max(worst_model_tokens, model_tokens)
    print(f"  chunk {c.index}: content_tokens={c.token_count:>3} "
          f"overlap={c.overlap_token_count:>2} model_tokens(+special)={model_tokens:>3}")

assert reconstruct(chunks) == doc, "reconstruction FAILED"
print("\nreconstruction: exact match OK")
print(f"worst-case model tokens: {worst_model_tokens} (must be <= 256)")
print("RESULT:", "PASS - no truncation" if worst_model_tokens <= 256 else "FAIL - would truncate")
