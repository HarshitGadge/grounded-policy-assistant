"""The Qdrant index as a RAGBench retriever.

RAGBench calls ``index(corpus)`` once with ``{doc_id: Document}``, then ``search`` or
``search_batch``. So the assistant's own retrieval can be scored on any BEIR dataset,
the regulation corpus included::

    ragbench run --data data/eval --retriever policy_assistant.retriever:QdrantRetriever \\
        --param model=BAAI/bge-small-en-v1.5

It searches an in-memory Qdrant by default, so a run leaves nothing behind.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .embed import DEFAULT_MODEL, Encoder, make_encoder
from .index import PolicyIndex, open_client


class QdrantRetriever:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        encoder: Encoder | None = None,
        path: str | None = None,
        url: str | None = None,
        collection: str = "ragbench",
        batch_size: int = 128,
    ):
        self.encoder = encoder or make_encoder(model)
        self.batch_size = batch_size
        self._index = PolicyIndex(open_client(path=path, url=url), self.encoder, collection)

    def index(self, corpus: Mapping[str, Any]) -> None:
        records = [
            {"_id": doc_id, "title": getattr(doc, "title", ""), "text": getattr(doc, "body", "")}
            for doc_id, doc in corpus.items()
        ]
        self._index.build(records, batch_size=self.batch_size)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: Sequence[str], k: int) -> list[list[tuple[str, float]]]:
        return [
            [(p.chunk_id, p.score) for p in hits]
            for hits in self._index.search_batch(list(queries), k)
        ]
