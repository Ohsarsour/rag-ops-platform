"""One-off probe: compare tiktoken token counts (what the chunker measures with)
against all-MiniLM-L6-v2's own tokenizer (what the embedding model actually uses).

Goal: find out whether a chunk sized to ~512 tiktoken tokens exceeds the
embedding model's maximum input length (so its tail would be silently
truncated and never represented in the vector).

This is a throwaway diagnostic, not part of the service. Run:
    python scripts/tokenizer_probe.py
"""
from __future__ import annotations

import tiktoken
from transformers import AutoTokenizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# A representative slice of Kubernetes-style technical prose. We build a chunk
# of ~512 tiktoken tokens (our chunker's target ceiling) and see how the
# embedding model's tokenizer counts the same text.
PARAGRAPH = (
    "A Pod is the smallest deployable unit of computing that you can create and "
    "manage in Kubernetes. A Pod is a group of one or more containers, with shared "
    "storage and network resources, and a specification for how to run the "
    "containers. A Pod's contents are always co-located and co-scheduled, and run "
    "in a shared context. A Pod models an application-specific logical host: it "
    "contains one or more application containers which are relatively tightly "
    "coupled. In non-cloud contexts, applications executed on the same physical or "
    "virtual machine are analogous to cloud applications executed on the same "
    "logical host. As well as application containers, a Pod can contain init "
    "containers that run during Pod startup. You can also inject ephemeral "
    "containers for debugging a running Pod. "
)

cl100k = tiktoken.get_encoding("cl100k_base")

# Grow the text until it measures ~512 tiktoken tokens, like a full chunk would.
text = ""
while len(cl100k.encode(text + PARAGRAPH)) <= 512:
    text += PARAGRAPH
# Top it up toward 512 with a final partial paragraph.
text += PARAGRAPH

tik_tokens = len(cl100k.encode(text))

hf_tok = AutoTokenizer.from_pretrained(MODEL)
# How the model itself counts (special tokens included, as during encoding).
hf_tokens_with_special = len(hf_tok.encode(text, add_special_tokens=True))
hf_tokens_no_special = len(hf_tok.encode(text, add_special_tokens=False))
model_max = hf_tok.model_max_length

print("=" * 60)
print(f"Sample text length:           {len(text)} characters")
print(f"tiktoken (cl100k_base):       {tik_tokens} tokens   <- chunker measures this")
print(f"MiniLM tokenizer (w/ special):{hf_tokens_with_special} tokens   <- model sees this")
print(f"MiniLM tokenizer (no special):{hf_tokens_no_special} tokens")
print(f"MiniLM model_max_length:      {model_max}")
print("=" * 60)

# all-MiniLM-L6-v2's effective max sequence length is 256 (set by
# sentence-transformers), even though the underlying BERT supports 512.
effective_cap = 256
if hf_tokens_with_special > effective_cap:
    lost = hf_tokens_with_special - effective_cap
    pct = 100 * lost / hf_tokens_with_special
    print(f"RESULT: a ~512 tiktoken-token chunk = {hf_tokens_with_special} MiniLM tokens,")
    print(f"        which EXCEEDS the 256 cap by {lost} tokens (~{pct:.0f}% truncated).")
else:
    print("RESULT: chunk fits within the 256-token cap; no truncation.")
