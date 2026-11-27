"""Index the corpus in Qdrant and search it.

Each chunk becomes one Qdrant point. Its id is derived from the chunk id, so rebuilding
the same corpus gives the same points, and its payload carries the citation and the text,
so a search result can be quoted and cited without going back to corpus.jsonl.

The collection records which encoder built it. Searching with a different encoder would
compare vectors from two different spaces, so :class:`PolicyIndex` refuses to.
"""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from .embed import Encoder

COLLECTION = "regulations"
# Fixed namespace, so a chunk id always maps to the same Qdrant point id.
_POINT_NAMESPACE = uuid.UUID("6f1d4b52-0c1e-4c8e-9a43-5d2b7f3a9e10")
_PAYLOAD_FIELDS = ("citation", "title", "text", "part", "section", "heading", "date")


@dataclass(frozen=True)
class Passage:
    """One search result: a chunk, its citation and its similarity to the question."""

    chunk_id: str
    score: float
    citation: str
    title: str
    text: str
    part: str
    section: str
    heading: str
    date: str


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def open_client(path: str | None = None, url: str | None = None, api_key: str | None = None):
    """A Qdrant server at ``url``, a local on-disk Qdrant at ``path``, or one in memory."""
    if url:
        return QdrantClient(url=url, api_key=api_key)
    if path and path != ":memory:":
        return QdrantClient(path=path)
    return QdrantClient(location=":memory:")


def collection_info(client: QdrantClient, collection: str = COLLECTION) -> dict | None:
    """The metadata a collection was built with, or None if there is no such collection."""
    if not client.collection_exists(collection):
        return None
    config = client.get_collection(collection).config
    meta = dict(config.metadata or {})
    meta.setdefault("dim", config.params.vectors.size)
    return meta


def passage_text(record: Mapping) -> str:
    """What gets embedded: the chunk's citation line, then its text (as RAGBench reads it)."""
    title = record.get("title") or ""
    text = record.get("text") or ""
    return f"{title}\n\n{text}".strip() if title else text.strip()


class PolicyIndex:
    def __init__(
        self,
        client: QdrantClient,
        encoder: Encoder,
        collection: str = COLLECTION,
        payload_indexes: bool = False,
    ):
        self.client = client
        self.encoder = encoder
        self.collection = collection
        # Local Qdrant ignores payload indexes (and warns), so only a server gets them.
        self.payload_indexes = payload_indexes

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def info(self) -> dict:
        """What the collection says about itself: the encoder, its size, the corpus date."""
        return collection_info(self.client, self.collection) or {}

    def build(
        self,
        records: Iterable[Mapping],
        batch_size: int = 128,
        metadata: Mapping | None = None,
    ) -> int:
        """Replace the collection with ``records`` (corpus.jsonl rows). Returns the count."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        rows = list(records)
        dupes = sorted(i for i, n in Counter(str(r["_id"]) for r in rows).items() if n > 1)
        if dupes:
            raise ValueError(f"chunk ids appear more than once: {dupes[:5]}")
        if self.exists():
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(
                size=self.encoder.dim, distance=models.Distance.COSINE
            ),
            metadata={
                "encoder": self.encoder.name,
                "dim": self.encoder.dim,
                "chunks": len(rows),
                **dict(metadata or {}),
            },
        )
        if self.payload_indexes:
            for field in ("part", "section"):
                self.client.create_payload_index(
                    self.collection, field, models.PayloadSchemaType.KEYWORD
                )
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            vectors = self.encoder.embed_documents([passage_text(r) for r in batch])
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(
                        id=point_id(str(r["_id"])),
                        vector=list(v),
                        payload={
                            "chunk_id": str(r["_id"]),
                            **{f: str(r.get(f) or "") for f in _PAYLOAD_FIELDS},
                        },
                    )
                    for r, v in zip(batch, vectors, strict=True)
                ],
                wait=True,
            )
        return self.count()

    def _check_encoder(self) -> None:
        if not self.exists():
            raise LookupError(f"no Qdrant collection {self.collection!r}; build the index first")
        built_with = self.info().get("encoder")
        if built_with and built_with != self.encoder.name:
            raise ValueError(
                f"collection {self.collection!r} was built with {built_with}, "
                f"but this search uses {self.encoder.name}"
            )

    @staticmethod
    def _filter(parts: Sequence[str] | None) -> models.Filter | None:
        if not parts:
            return None
        return models.Filter(
            must=[models.FieldCondition(key="part", match=models.MatchAny(any=list(parts)))]
        )

    @staticmethod
    def _passage(point) -> Passage:
        p = point.payload or {}
        return Passage(
            chunk_id=p.get("chunk_id", ""),
            score=float(point.score),
            **{f: p.get(f, "") for f in _PAYLOAD_FIELDS},
        )

    def search(self, query: str, k: int = 5, parts: Sequence[str] | None = None) -> list[Passage]:
        return self.search_batch([query], k, parts)[0]

    def search_batch(
        self, queries: Sequence[str], k: int = 5, parts: Sequence[str] | None = None
    ) -> list[list[Passage]]:
        """The ``k`` closest chunks to each query, best first, optionally only in ``parts``."""
        if k < 1:
            raise ValueError("k must be at least 1")
        if not queries:
            return []
        self._check_encoder()
        vectors = self.encoder.embed_queries(list(queries))
        flt = self._filter(parts)
        responses = self.client.query_batch_points(
            self.collection,
            requests=[
                models.QueryRequest(query=list(v), limit=k, filter=flt, with_payload=True)
                for v in vectors
            ],
        )
        return [[self._passage(p) for p in r.points] for r in responses]
