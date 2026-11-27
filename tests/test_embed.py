import math
import sys
import types

import pytest

from policy_assistant import embed
from policy_assistant.embed import FastEmbedEncoder, HashingEncoder, make_encoder


def test_hashing_is_deterministic_and_normalised():
    enc = HashingEncoder(64)
    a, b = enc.embed_documents(["Access device means a card", "Access device means a card"])
    assert a == b
    assert len(a) == 64
    assert math.isclose(sum(x * x for x in a), 1.0)
    assert enc.embed_queries(["Access device means a card"])[0] == a


def test_hashing_shares_words_not_unrelated_text():
    enc = HashingEncoder(512)
    q = enc.embed_queries(["unauthorized transfer liability"])[0]
    near, far = enc.embed_documents(
        ["Liability for an unauthorized transfer", "Record retention for two years"]
    )

    def cos(u, v):
        return sum(x * y for x, y in zip(u, v, strict=True))

    assert cos(q, near) > cos(q, far)


def test_hashing_empty_text_is_a_zero_vector():
    assert HashingEncoder(16).embed_documents(["  ...  "])[0] == [0.0] * 16


def test_hashing_rejects_tiny_dims():
    with pytest.raises(ValueError):
        HashingEncoder(4)


def test_make_encoder_reads_hashing_names():
    enc = make_encoder("hashing-128")
    assert isinstance(enc, HashingEncoder)
    assert enc.dim == 128
    assert enc.name == "hashing-128"


@pytest.fixture
def fake_fastembed(monkeypatch):
    """A stand-in for fastembed that records which of its methods were called."""
    calls = []

    class TextEmbedding:
        embedding_size = 3

        def __init__(self, model_name, cache_dir=None, threads=None):
            calls.append(("init", model_name))

        def passage_embed(self, texts, batch_size=256):
            calls.append(("passage", list(texts), batch_size))
            return [[1, 0, 0] for _ in texts]

        def query_embed(self, texts):
            calls.append(("query", list(texts)))
            return [[0, 1, 0] for _ in texts]

    module = types.ModuleType("fastembed")
    module.TextEmbedding = TextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", module)
    return calls


def test_fastembed_uses_passage_and_query_embedding(fake_fastembed):
    enc = make_encoder("some/model", batch_size=8)
    assert isinstance(enc, FastEmbedEncoder)
    assert (enc.name, enc.dim) == ("some/model", 3)
    assert enc.embed_documents(["a", "b"]) == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert enc.embed_queries(["q"]) == [[0.0, 1.0, 0.0]]
    assert fake_fastembed == [
        ("init", "some/model"),
        ("passage", ["a", "b"], 8),
        ("query", ["q"]),
    ]


def test_fastembed_missing_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(ImportError, match=r"grounded-policy-assistant\[fastembed\]"):
        FastEmbedEncoder()


def test_default_model_is_bge_small():
    assert embed.DEFAULT_MODEL == "BAAI/bge-small-en-v1.5"
