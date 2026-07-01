"""Splits a document into small, overlapping pieces ("chunks") for the RAG pipeline.

Why chunking exists (read this first if you're new)
---------------------------------------------------
The pipeline answers questions by finding relevant text in our documents and
handing it to an AI model. Two practical limits force us to cut documents into
small pieces before we store them:

1. The AI model can only read so much text at once, and reading more costs more
   time and memory. We can't paste a whole 50-page document into it per question.
2. Search is far more precise on small pieces. If a whole document is stored as
   one blob, a search can only say "this big document is sort of relevant." If
   the same document is stored as 200 small pieces, the search can return the one
   paragraph that actually answers the question.

So "chunking" just means chopping each document into small, self-contained pieces
of text. This module does exactly that. It is plain text-cutting logic: no AI
model, no database, no network calls. Keeping it that simple means it can be
imported and tested entirely on its own.

What a "token" is, and why we count with the embedding model's tokenizer
------------------------------------------------------------------------
AI tools don't measure text in characters or words; they measure it in *tokens*.
A token is roughly a word-piece. Crucially, *different models tokenize
differently* and each embedding model can only read a fixed number of tokens per
input - anything longer is silently truncated (dropped) before it becomes a
vector. Our embedding model, ``all-MiniLM-L6-v2``, truncates anything past **256
word-pieces** (including the 2 special [CLS]/[SEP] markers it adds).

So we must size chunks in *that model's* own tokens, not some other tokenizer's.
We measure with the model's tokenizer (via ``transformers.AutoTokenizer``) and
keep every chunk comfortably under 256 so the whole chunk is actually embedded.
(An earlier version counted with OpenAI's ``tiktoken`` and targeted 512 tokens;
testing showed a "512-token" chunk was ~634 MiniLM tokens, so ~60% of every full
chunk was being truncated away before embedding. See docs/learning-notes.md.)

The rules each chunk follows
----------------------------
* Target size: up to 246 tokens per chunk (246 content tokens + 2 special markers
  = 248, which stays under MiniLM's 256 cap so nothing is truncated). Every chunk
  except the last also stays at or above 200 tokens, so we don't emit lots of tiny
  scraps. The last chunk is whatever is left over (1..246 tokens).
* Overlap: each chunk after the first repeats the last 40..50 tokens of the
  previous chunk at its start. Think of roof shingles overlapping. This stops an
  idea that straddles a boundary from falling through the crack between two chunks.
* Natural breakpoints: we prefer to cut at paragraph breaks, then line breaks,
  then sentence ends, then spaces, and only split mid-word as a last resort. The
  separators we try, largest break first, are ``["\\n\\n", "\\n", ". ", " ", ""]``.
* Whole-document case: a document of 246 tokens or fewer becomes a single chunk
  containing all of its text.
* Ordering: chunks are numbered 0, 1, 2, ... in document order, with no gaps.
* No data loss: if you stitch the chunks back together (dropping the repeated
  overlap each chunk carries), you get the original text back exactly. Every chunk
  records how many leading characters are overlap so this stitching is unambiguous.
* Empty or whitespace-only input produces zero chunks.

How it works under the hood
---------------------------
The "stitch back to the original exactly" guarantee is the trickiest part, so we
get it for free by tracking, for every chunk, the character positions
``[start_char, end_char)`` it occupies in the *original* text:

* Chunk 0 covers ``[0, e0)``.
* Each later chunk starts a little *before* the previous chunk ended (that's the
  overlap). The repeated characters at its start are exactly the tail of the
  previous chunk; the chunk's unique "body" is the slice that comes after the
  previous chunk's end.

Because the bodies are back-to-back, non-overlapping slices that together cover
the whole text, dropping each chunk's repeated prefix and concatenating the
bodies reproduces the original exactly.

Sizing is done in three steps:

1. Cut the text into small "pieces" (each <= MAX_PIECE tokens) whose
   concatenation still equals the original text exactly. We cut at the largest
   natural separator that keeps pieces small, going finer (down to single
   characters) only when a stretch of text has no usable break.
2. Build each chunk by packing whole pieces together until adding one more would
   exceed 246 tokens. Because the pieces are small, a chunk can never stop far
   below 246, so non-final chunks always land at or above 200 tokens.
3. Start the next chunk by walking backward from the current chunk's end until the
   repeated tail measures 40..50 tokens. That repeated tail is the overlap.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from transformers import PreTrainedTokenizerBase

# --- Size limits, all measured in the embedding model's tokens ---------------

#: Largest a chunk may be, in content tokens. With the 2 special markers MiniLM
#: adds at embed time this is 248 tokens, which stays under the model's 256 cap
#: so the entire chunk is embedded rather than truncated.
MAX_CHUNK_TOKENS = 246
#: Smallest a chunk may be, except for the final leftover chunk.
MIN_CHUNK_TOKENS = 200
#: How much each chunk repeats from the previous one (the overlap window).
OVERLAP_MIN_TOKENS = 40
OVERLAP_MAX_TOKENS = 50

#: Largest a single "piece" may be, in tokens. We keep this below the
#: 246 - 200 = 46 token gap on purpose. When we stop packing pieces into a
#: chunk, the piece we rejected was small (at most ~MAX_PIECE tokens), so the
#: chunk we already built must be within that small amount of the 246 ceiling -
#: which keeps every non-final chunk at or above 200 tokens.
MAX_PIECE = 35

#: Breakpoints to try, largest natural boundary first:
#: paragraph -> line -> sentence -> word -> single character.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

#: The tokenizer to size chunks with. Must match the embedding model so our
#: token counts are in the same units the model uses. Configurable so it stays
#: in step with the embedding service if the model is ever swapped.
TOKENIZER_NAME = os.environ.get(
    "CHUNK_TOKENIZER", "sentence-transformers/all-MiniLM-L6-v2"
)

# Loading the tokenizer is relatively expensive and needs the model files, so we
# load it once on first use and reuse it for every call.
_tokenizer: "Optional[PreTrainedTokenizerBase]" = None


def _get_tokenizer() -> "PreTrainedTokenizerBase":
    """Load (once) and return the embedding model's tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        # Imported lazily so this module can be imported without the dependency
        # present (e.g. for tests that don't exercise real token counting).
        from transformers import AutoTokenizer
        from transformers.utils import logging as hf_logging

        # We deliberately measure whole documents (longer than the model's input
        # limit) to decide how to split them, so silence the tokenizer's
        # "sequence longer than maximum" warning - it is expected here and would
        # otherwise log once per ingested document.
        hf_logging.set_verbosity_error()
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _tokenizer


class ChunkingError(Exception):
    """Raised when chunking fails or produces output that breaks the size rules.

    The Ingestion service treats this as a hard failure: it stores nothing for
    the document and returns an error, so a bad split never leaves half a
    document in the database.
    """


@dataclass(frozen=True)
class Chunk:
    """One produced piece of a document.

    Attributes:
        index: Position in document order, starting at 0.
        content: The chunk's text, including the overlap repeated at its start.
        token_count: How many tokens ``content`` is.
        overlap_token_count: How many tokens at the start are repeated from the
            previous chunk (0 for the first chunk).
        overlap_char_count: How many leading characters of ``content`` are the
            repeated overlap. Dropping ``content[:overlap_char_count]`` leaves the
            chunk's unique body; concatenating those bodies in order rebuilds the
            original document.
        start_char: Where ``content`` begins in the original text (inclusive).
        end_char: Where ``content`` ends in the original text (exclusive).
    """

    index: int
    content: str
    token_count: int
    overlap_token_count: int
    overlap_char_count: int
    start_char: int
    end_char: int


def token_length(text: str) -> int:
    """Return how many tokens ``text`` is, in the embedding model's units.

    Counts content word-pieces only (``tokenize`` excludes the [CLS]/[SEP]
    special markers), which is the number we compare against the chunk-size
    limits. The model adds those 2 markers itself at embed time, so a 246-token
    chunk becomes 248 tokens for the model, still under its 256 cap.
    """
    if not text:
        return 0
    return len(_get_tokenizer().tokenize(text))


def _split_keep_separator(text: str, separator: str) -> List[str]:
    """Split ``text`` on ``separator`` but keep the separator attached.

    The separator stays glued to the end of the fragment before it, so joining
    the returned list back together reproduces ``text`` exactly. An empty
    separator means "split into individual characters."
    """
    if separator == "":
        return list(text)
    parts = text.split(separator)
    result: List[str] = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            result.append(part + separator)
        elif part:  # a trailing empty fragment means text ended with the separator
            result.append(part)
    return result


def _group_chars(text: str) -> List[str]:
    """Cut ``text`` into pieces of at most MAX_PIECE tokens, character by character.

    This is the last resort for a stretch of text that has no usable separator
    but is still too big (for example one very long unbroken string).
    """
    pieces: List[str] = []
    current = ""
    for ch in text:
        candidate = current + ch
        if current and token_length(candidate) > MAX_PIECE:
            pieces.append(current)
            current = ch
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _split_into_pieces(text: str, separators: List[str]) -> List[str]:
    """Cut ``text`` into small pieces (each <= MAX_PIECE tokens).

    Joining the returned pieces reproduces ``text`` exactly. We split at the
    largest separator that applies and only fall back to finer separators (and
    ultimately single characters) for stretches that are still too big.
    """
    if token_length(text) <= MAX_PIECE:
        return [text] if text else []

    if not separators:
        return _group_chars(text)

    separator, rest = separators[0], separators[1:]

    if separator == "":
        return _group_chars(text)

    parts = _split_keep_separator(text, separator)
    if len(parts) <= 1:
        # This separator isn't in the text; try the next finer one.
        return _split_into_pieces(text, rest)

    pieces: List[str] = []
    for part in parts:
        if not part:
            continue
        if token_length(part) <= MAX_PIECE:
            pieces.append(part)
        else:
            pieces.extend(_split_into_pieces(part, rest))
    return pieces


def _piece_boundaries(pieces: List[str]) -> List[int]:
    """Return the character position at the end of each piece.

    These positions are the only places a chunk is allowed to *end*. The list
    always ends with the full length of the text.
    """
    boundaries: List[int] = []
    offset = 0
    for piece in pieces:
        offset += len(piece)
        boundaries.append(offset)
    return boundaries


def _largest_end_within_budget(text: str, start: int, boundaries: List[int]) -> int:
    """Find the furthest piece boundary we can reach from ``start`` within 246 tokens.

    A longer slice always has at least as many tokens as a shorter one, so as we
    consider boundaries further to the right the token count only goes up. That
    lets us binary-search for the furthest boundary whose slice still fits in 246
    tokens. There is always at least one reachable boundary because every piece is
    small enough to fit under the limit on its own.
    """
    # Find the first boundary that lies strictly after start.
    lo, hi = 0, len(boundaries) - 1
    first = len(boundaries)
    while lo <= hi:
        mid = (lo + hi) // 2
        if boundaries[mid] > start:
            first = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if first == len(boundaries):
        # Shouldn't happen: start is already past every boundary.
        raise ChunkingError("no chunk boundary available beyond start offset")

    # From there, pick the furthest boundary whose slice still fits in 246 tokens.
    best = boundaries[first]  # always take at least one piece
    lo, hi = first, len(boundaries) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        end = boundaries[mid]
        if token_length(text[start:end]) <= MAX_CHUNK_TOKENS:
            best = end
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _overlap_start(text: str, end: int, lower_bound: int) -> int:
    """Find where the next chunk should start so it repeats 40..50 tokens.

    Walks backward from ``end``, pulling in as many characters as possible while
    the repeated tail stays at or below the overlap limit, then returns that
    position. ``lower_bound`` is where the current chunk started, so the overlap
    never reaches further back than the current chunk's own text.
    """
    start = end
    while start > lower_bound and token_length(text[start - 1:end]) <= OVERLAP_MAX_TOKENS:
        start -= 1
    return start


def chunk_text(text: str) -> List[Chunk]:
    """Split ``text`` into overlapping, token-sized chunks.

    Args:
        text: The raw document text.

    Returns:
        The chunks in document order. Empty or whitespace-only input returns an
        empty list.

    Raises:
        ChunkingError: If the result somehow breaks the size or overlap rules.
            This is a safety net; it shouldn't happen with valid input.
    """
    # Empty or whitespace-only content produces no chunks.
    if not text or not text.strip():
        return []

    total_tokens = token_length(text)

    # A document of 246 tokens or fewer is a single chunk with all of its text.
    if total_tokens <= MAX_CHUNK_TOKENS:
        chunk = Chunk(
            index=0,
            content=text,
            token_count=total_tokens,
            overlap_token_count=0,
            overlap_char_count=0,
            start_char=0,
            end_char=len(text),
        )
        return [chunk]

    pieces = _split_into_pieces(text, SEPARATORS)
    if not pieces:  # pragma: no cover - the strip() check above already guards this
        return []
    boundaries = _piece_boundaries(pieces)

    chunks: List[Chunk] = []
    n = len(text)
    start = 0
    prev_start = 0
    prev_end = 0
    index = 0

    while start < n:
        end = _largest_end_within_budget(text, start, boundaries)

        if index == 0:
            overlap_chars = 0
        else:
            # The overlap is text[start:prev_end]: the tail of the previous chunk
            # repeated at the start of this one.
            overlap_chars = prev_end - start

        content = text[start:end]
        chunks.append(
            Chunk(
                index=index,
                content=content,
                token_count=token_length(content),
                overlap_token_count=token_length(text[start:prev_end]) if index > 0 else 0,
                overlap_char_count=overlap_chars,
                start_char=start,
                end_char=end,
            )
        )

        if end >= n:
            break

        # Start the next chunk inside this one so the two overlap.
        next_start = _overlap_start(text, end, lower_bound=start)
        if next_start <= start:  # pragma: no cover - guards against an infinite loop
            raise ChunkingError("chunking failed to make forward progress")

        prev_start = start
        prev_end = end
        start = next_start
        index += 1

    _validate(chunks)
    return chunks


def _validate(chunks: List[Chunk]) -> None:
    """Double-check the chunks obey the size rules; raise if any don't.

    This is a safety net so a subtle bug can never silently store
    wrong-sized chunks.
    """
    if not chunks:
        return

    last = len(chunks) - 1
    for i, c in enumerate(chunks):
        # Indices must be 0, 1, 2, ... with no gaps.
        if c.index != i:
            raise ChunkingError(f"chunk index {c.index} out of order at position {i}")

        # Size limits (the final leftover chunk is allowed to be small).
        if i < last:
            if not (MIN_CHUNK_TOKENS <= c.token_count <= MAX_CHUNK_TOKENS):
                raise ChunkingError(
                    f"chunk {i} has {c.token_count} tokens, outside "
                    f"[{MIN_CHUNK_TOKENS}, {MAX_CHUNK_TOKENS}]"
                )
        else:
            if not (1 <= c.token_count <= MAX_CHUNK_TOKENS):
                raise ChunkingError(
                    f"final chunk has {c.token_count} tokens, outside "
                    f"[1, {MAX_CHUNK_TOKENS}]"
                )

        # Overlap window, for every chunk after the first.
        if i > 0 and not (OVERLAP_MIN_TOKENS <= c.overlap_token_count <= OVERLAP_MAX_TOKENS):
            raise ChunkingError(
                f"chunk {i} overlap is {c.overlap_token_count} tokens, outside "
                f"[{OVERLAP_MIN_TOKENS}, {OVERLAP_MAX_TOKENS}]"
            )


def reconstruct(chunks: List[Chunk]) -> str:
    """Rebuild the original text from its chunks.

    Drops each chunk's repeated overlap and joins the unique bodies in order.
    Handy for callers and tests that want to prove no text was lost.
    """
    parts: List[str] = []
    for c in sorted(chunks, key=lambda c: c.index):
        parts.append(c.content[c.overlap_char_count:])
    return "".join(parts)
