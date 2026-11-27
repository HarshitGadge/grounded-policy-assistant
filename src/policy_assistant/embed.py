"""Text encoders for the index.

An encoder turns passages and questions into vectors. It has a ``name`` (stored with
the index, so a search uses the same encoder that built it), a ``dim``, and two
methods, ``embed_documents`` and ``embed_queries``, the shape RAGBench's dense
retriever also takes.

* :class:`FastEmbedEncoder` runs a `fastembed <https://github.com/qdrant/fastembed>`_
  model on the CPU (ONNX). Install it with ``pip install 'grounded-policy-assistant[fastembed]'``.
  The default is ``BAAI/bge-small-en-v1.5`` (384 dimensions).
* :class:`HashingEncoder` hashes words into a fixed number of buckets. It needs no model
  download, so the offline tests use it. It only matches shared words.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
HASHING_PREFIX = "hashing-"

_WORD = re.compile(r"[a-z0-9]+")


class Encoder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedEncoder:
    """A fastembed text model. Passages and queries go through the model's own
    ``passage_embed`` and ``query_embed``, so models that want a query instruction get it."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        batch_size: int = 64,
        cache_dir: str | None = None,
        threads: int | None = None,
    ):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "FastEmbedEncoder needs fastembed: "
                "pip install 'grounded-policy-assistant[fastembed]'"
            ) from exc
        self.name = model
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model, cache_dir=cache_dir, threads=threads)
        size = getattr(self._model, "embedding_size", None)
        self.dim = int(size) if size else len(self.embed_queries(["dimension probe"])[0])

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.passage_embed(list(texts), batch_size=self.batch_size)
        return [[float(x) for x in v] for v in vectors]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._model.query_embed(list(texts))]


class HashingEncoder:
    """Signed feature hashing of lower-cased words and word pairs, then L2-normalised.

    Deterministic across machines and Python versions (it uses BLAKE2b, not ``hash``).
    """

    def __init__(self, dim: int = 512):
        if dim < 8:
            raise ValueError(f"dim must be at least 8, got {dim}")
        self.dim = dim
        self.name = f"{HASHING_PREFIX}{dim}"

    def _vector(self, text: str) -> list[float]:
        words = _WORD.findall(text.lower())
        features = words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]
        vec = [0.0] * self.dim
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            vec[bucket] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


def make_encoder(name: str = DEFAULT_MODEL, **kwargs) -> Encoder:
    """``hashing-512`` gives a :class:`HashingEncoder`; anything else is a fastembed model."""
    if name.startswith(HASHING_PREFIX):
        return HashingEncoder(int(name[len(HASHING_PREFIX) :]))
    return FastEmbedEncoder(name, **kwargs)
