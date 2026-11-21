"""Shared, fully deterministic mock decoding helpers.

Both mock nodes import exactly this module, so a healthy continuation
reproduces the same token stream byte-for-byte. Nothing here uses random
number generators, wall clocks or machine identifiers: the token sequence is
a pure function of (prompt, continuation offset).
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache


def _digest(prompt: str, index: int) -> bytes:
    """Stable per-position hash derived from the prompt and token index."""
    h = hashlib.sha256()
    h.update(b"mock-llm/v1:")
    h.update(prompt.encode("utf-8"))
    h.update(b":")
    h.update(str(index).encode("ascii"))
    return h.digest()


@lru_cache(maxsize=8)
def load_vocab() -> tuple[str, ...]:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "data", "vocab.json")
    with open(path, "rb") as fh:
        return tuple(json.load(fh))


def token_id(prompt: str, index: int) -> int:
    """Deterministic token id (index into the shared vocabulary)."""
    vocab = load_vocab()
    return int.from_bytes(_digest(prompt, index)[:4], "big") % len(vocab)


def token_text(prompt: str, index: int) -> str:
    """Deterministic token text. Joining these with spaces yields the reply."""
    vocab = load_vocab()
    return vocab[token_id(prompt, index)]


def expected_token_ids(prompt: str, start: int, end: int) -> list[int]:
    return [token_id(prompt, i) for i in range(start, end)]
