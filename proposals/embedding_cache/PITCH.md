# Persistent embedding cache

`cached_embed(text, embed_fn, store_path, model_tag="")` wraps any embed
function with a content-hash-keyed on-disk cache (a plain JSON file, keyed
by `sha256(model_tag + text)`), so the same text never gets re-embedded
across retrieval calls or process restarts. It's valuable because
`embeddings.embed` is the dominant per-call cost in `retriever.retrieve()`
(a network round trip to Ollama's GPU embedding model), and lesson/query
text is frequently repeated — recall.py, tune_min_sim.py, and repeated
solver retries all re-embed identical strings today.

To integrate: pass `lambda t: embedding_cache.cached_embed(t, embeddings.embed, "memory_cache.json", model_tag=embeddings.EMBED_MODEL)`
as the `embed_fn` argument to `retriever.retrieve()` / `retriever.semantic_search()`
(both already accept an injectable `embed_fn`, no signature change needed).
Soft-fails (embed_fn returning `None`) are never cached, so a temporarily
down Ollama still retries on the next call instead of poisoning the cache.
