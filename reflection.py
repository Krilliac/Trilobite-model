"""Distill reusable lessons from good outcomes; dedup by embedding similarity."""
import embeddings
import memory_store

DUP_THRESHOLD = 0.92
DISTILL_SYSTEM = (
    "You distill a single reusable coding lesson. Output one short imperative "
    "sentence, no preamble, no markdown."
)


def distill(task, response, signal, offload_fn):
    prompt = (
        "A coding task was completed with outcome '%s'.\n\n"
        "TASK:\n%s\n\nSOLUTION:\n%s\n\n"
        "Write ONE short imperative lesson (max 25 words) capturing the reusable "
        "insight for similar future tasks. No preamble." % (signal, task, response)
    )
    text = offload_fn(
        prompt=prompt, tier="fast", system=DISTILL_SYSTEM,
        temperature=0.0, num_predict=60,
    )
    return (text or "").strip()


def is_duplicate(new_emb, conn, threshold=DUP_THRESHOLD):
    if new_emb is None:
        return False
    for les in memory_store.all_lessons(conn):
        emb = les["embedding"]
        if emb and embeddings.cosine(new_emb, embeddings.from_blob(emb)) >= threshold:
            return True
    return False


def maybe_add_lesson(conn, interaction_id, task, response, signal, offload_fn,
                     embed_fn=embeddings.embed, id_fn=memory_store.new_id):
    if memory_store.lesson_exists_for_interaction(conn, interaction_id):
        return None
    text = distill(task, response, signal, offload_fn)
    if not text:
        return None
    emb = embed_fn(text)
    if is_duplicate(emb, conn):
        return None
    lesson_id = id_fn()
    blob = embeddings.to_blob(emb) if emb else None
    memory_store.add_lesson(conn, lesson_id, text, blob, interaction_id)
    return lesson_id
