import json
import os

import embedding_cache as ec


class CountingEmbedder:
    """Stub embed_fn that counts calls and returns a deterministic vector.

    No GPU/network involved: the "vector" is derived from the text length
    so equal inputs always produce equal outputs, and calls are counted so
    tests can assert the cache actually avoided recomputation.
    """

    def __init__(self, vec=None):
        self.calls = 0
        self._vec = vec

    def __call__(self, text):
        self.calls += 1
        if self._vec is not None:
            return list(self._vec)
        return [float(len(text)), float(sum(map(ord, text)) % 97)]


class NoneEmbedder:
    """Stub embed_fn that always soft-fails, like embeddings.embed does
    when Ollama is unreachable."""

    def __init__(self):
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        return None


def test_cache_key_deterministic_and_text_sensitive():
    k1 = ec.cache_key("hello")
    k2 = ec.cache_key("hello")
    k3 = ec.cache_key("world")
    assert k1 == k2
    assert k1 != k3


def test_cache_key_sensitive_to_model_tag():
    k1 = ec.cache_key("hello", model_tag="model-a")
    k2 = ec.cache_key("hello", model_tag="model-b")
    assert k1 != k2


def test_cache_hit_avoids_recompute(tmp_path):
    store = str(tmp_path / "cache.json")
    stub = CountingEmbedder()

    v1 = ec.cached_embed("some lesson text", stub, store)
    v2 = ec.cached_embed("some lesson text", stub, store)

    assert v1 == v2
    assert stub.calls == 1  # second call was served entirely from cache


def test_different_text_both_computed(tmp_path):
    store = str(tmp_path / "cache.json")
    stub = CountingEmbedder()

    v1 = ec.cached_embed("text one", stub, store)
    v2 = ec.cached_embed("text two", stub, store)

    assert v1 != v2
    assert stub.calls == 2


def test_persists_across_simulated_process_restart(tmp_path):
    store = str(tmp_path / "cache.json")
    stub = CountingEmbedder()

    v1 = ec.cached_embed("persist me", stub, store)
    assert stub.calls == 1

    # Simulate a fresh process: drop the in-memory cache, keep the on-disk
    # file. A truly persistent cache must still hit without reloading via
    # cached_embed's in-memory shortcut.
    ec.forget_process_cache(store)
    assert os.path.exists(store)

    v2 = ec.cached_embed("persist me", stub, store)
    assert v2 == v1
    assert stub.calls == 1  # no recompute after "restart"


def test_store_file_is_plain_json_dict(tmp_path):
    store = str(tmp_path / "cache.json")
    stub = CountingEmbedder(vec=[1.0, 2.0, 3.0])

    ec.cached_embed("hi", stub, store)

    with open(store, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert len(data) == 1
    key = ec.cache_key("hi")
    assert data[key] == [1.0, 2.0, 3.0]


def test_model_tag_isolates_cache_entries(tmp_path):
    store = str(tmp_path / "cache.json")
    stub = CountingEmbedder()

    ec.cached_embed("same text", stub, store, model_tag="model-a")
    ec.cached_embed("same text", stub, store, model_tag="model-b")

    # Different models must not share a cache slot even for identical text.
    assert stub.calls == 2


def test_soft_fail_is_not_cached_and_retries_next_call(tmp_path):
    store = str(tmp_path / "cache.json")
    stub = NoneEmbedder()

    r1 = ec.cached_embed("unreachable", stub, store)
    r2 = ec.cached_embed("unreachable", stub, store)

    assert r1 is None
    assert r2 is None
    assert stub.calls == 2  # each call retried embed_fn; nothing was cached


def test_cache_stats_reports_entry_count(tmp_path):
    store = str(tmp_path / "cache.json")
    stub = CountingEmbedder()

    assert ec.cache_stats(store) == {"entries": 0}

    ec.cached_embed("a", stub, store)
    ec.cached_embed("b", stub, store)
    ec.cached_embed("a", stub, store)  # repeat, should not add a new entry

    assert ec.cache_stats(store) == {"entries": 2}


def test_clear_cache_removes_file_and_memory(tmp_path):
    store = str(tmp_path / "cache.json")
    stub = CountingEmbedder()

    ec.cached_embed("a", stub, store)
    assert os.path.exists(store)

    ec.clear_cache(store)

    assert not os.path.exists(store)
    assert ec.cache_stats(store) == {"entries": 0}

    # After clearing, the same text must be recomputed.
    ec.cached_embed("a", stub, store)
    assert stub.calls == 2


def test_returned_vector_is_independent_copy(tmp_path):
    """Mutating a vector returned from the cache must not corrupt the
    cached value for the next caller."""
    store = str(tmp_path / "cache.json")
    stub = CountingEmbedder(vec=[1.0, 2.0])

    v1 = ec.cached_embed("shared", stub, store)
    v1.append(999.0)

    v2 = ec.cached_embed("shared", stub, store)
    assert v2 == [1.0, 2.0]


def test_integrates_with_real_embeddings_module_signature(tmp_path):
    """cached_embed's embed_fn contract matches embeddings.embed(text) ->
    list|None, so it drops in as retriever.retrieve's embed_fn without
    any adapter."""
    import embeddings

    store = str(tmp_path / "cache.json")
    calls = []

    def fake_ollama_embed(text, timeout=30):
        calls.append(text)
        return [0.1, 0.2]

    def embed_fn(text):
        return fake_ollama_embed(text)

    assert embeddings.embed.__code__.co_varnames[0] == "text"

    v1 = ec.cached_embed("query text", embed_fn, store, model_tag=embeddings.EMBED_MODEL)
    v2 = ec.cached_embed("query text", embed_fn, store, model_tag=embeddings.EMBED_MODEL)

    assert v1 == v2 == [0.1, 0.2]
    assert len(calls) == 1
