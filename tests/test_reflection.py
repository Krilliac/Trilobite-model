# tests/test_reflection.py
import embeddings as e
import memory_store as ms
import reflection


def _off(**kw):
    # stub offload: echoes a fixed lesson regardless of prompt
    return "  Release the lock in a finally block.  "


def test_distill_strips_and_returns_text():
    out = reflection.distill("task", "resp", "tests_passed", _off)
    assert out == "Release the lock in a finally block."


def test_maybe_add_lesson_writes_one_lesson():
    c = ms.connect(":memory:")
    lid = reflection.maybe_add_lesson(
        c, "i1", "task", "resp", "tests_passed",
        offload_fn=_off, embed_fn=lambda t: [1.0, 0.0],
    )
    assert lid is not None
    assert ms.get_lesson_text(c, lid) == "Release the lock in a finally block."


def test_maybe_add_lesson_dedupes_near_duplicate():
    c = ms.connect(":memory:")
    ms.add_lesson(c, "existing", "Release the lock in a finally block.",
                  e.to_blob([1.0, 0.0]), "i0")
    lid = reflection.maybe_add_lesson(
        c, "i1", "task", "resp", "tests_passed",
        offload_fn=_off, embed_fn=lambda t: [1.0, 0.0],  # identical vector -> dup
    )
    assert lid is None
    assert len(ms.all_lessons(c)) == 1


def test_maybe_add_lesson_skips_empty_distill():
    c = ms.connect(":memory:")
    lid = reflection.maybe_add_lesson(
        c, "i1", "task", "resp", "tests_passed",
        offload_fn=lambda **kw: "   ", embed_fn=lambda t: [1.0, 0.0],
    )
    assert lid is None
