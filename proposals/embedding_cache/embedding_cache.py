"""Content-hash-keyed persistent cache for embedding vectors. Stdlib only.

Wraps any embed_fn(text) -> vector so repeated calls with the same text
(across retrieval calls, and across process restarts) hit a small on-disk
JSON file instead of recomputing the embedding. Useful in front of
embeddings.embed, whose GPU/Ollama round trip is the dominant cost of
retriever.retrieve() for lessons that get queried or re-embedded often.

Cache key = sha256(model_tag + "\\0" + text). model_tag defaults to "" but
callers should pass the embed model name (e.g. embeddings.EMBED_MODEL) so
switching models can't silently return a stale vector from a different one.
"""
import hashlib
import json
import os
import threading

_LOCK = threading.Lock()
_MEM_CACHE = {}  # store_path -> {key: vector} loaded from disk, kept in sync


def cache_key(text, model_tag=""):
    h = hashlib.sha256()
    h.update(model_tag.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _load(store_path):
    if store_path in _MEM_CACHE:
        return _MEM_CACHE[store_path]
    data = {}
    if os.path.exists(store_path):
        try:
            with open(store_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
        except (ValueError, OSError):
            data = {}
    _MEM_CACHE[store_path] = data
    return data


def _save(store_path, data):
    parent = os.path.dirname(store_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = store_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp_path, store_path)


def cached_embed(text, embed_fn, store_path, model_tag=""):
    """Return embed_fn(text)'s vector, transparently cached at store_path.

    On a cache hit, embed_fn is never called. On a miss, embed_fn(text) is
    called; if it returns a truthy (non-empty) vector, the vector is
    persisted and returned. If embed_fn soft-fails (returns None or an
    empty vector, as embeddings.embed does when Ollama is unreachable),
    nothing is cached, so the next call retries embed_fn instead of
    permanently caching a failure.
    """
    key = cache_key(text, model_tag)
    with _LOCK:
        data = _load(store_path)
        if key in data:
            return list(data[key])

    vec = embed_fn(text)

    if vec:
        with _LOCK:
            data = _load(store_path)
            data[key] = list(vec)
            _save(store_path, data)
    return vec


def cache_stats(store_path):
    """Return {"entries": n} describing the on-disk cache at store_path."""
    with _LOCK:
        data = _load(store_path)
        return {"entries": len(data)}


def clear_cache(store_path):
    """Drop the in-memory and on-disk cache at store_path."""
    with _LOCK:
        _MEM_CACHE[store_path] = {}
        if os.path.exists(store_path):
            try:
                os.remove(store_path)
            except OSError:
                pass


def forget_process_cache(store_path):
    """Evict store_path from the in-process memory cache only.

    The on-disk file is left untouched. Used to simulate a fresh process
    reading the cache back from disk (e.g. in tests), without deleting the
    persisted data.
    """
    with _LOCK:
        _MEM_CACHE.pop(store_path, None)
