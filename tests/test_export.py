import memory_store as ms
import export_training_data as etd


def _conn():
    return ms.connect(":memory:")


def test_build_examples_returns_good_pairs_in_chat_shape():
    c = _conn()
    ms.log_interaction(c, "a", "task A", "", "resp A", "code")
    ms.log_interaction(c, "b", "task B", "", "resp B", "code")
    ms.log_interaction(c, "bad", "task bad", "", "resp bad", "code")
    ms.record_outcome_row(c, "a", "tests_passed", 1.0)
    ms.record_outcome_row(c, "b", "compiled", 0.7)
    ms.record_outcome_row(c, "bad", "failed", -1.0)

    examples = etd.build_examples(c)

    assert len(examples) == 2
    tasks = {ex["messages"][0]["content"] for ex in examples}
    assert tasks == {"task A", "task B"}
    for ex in examples:
        assert ex["messages"][0]["role"] == "user"
        assert ex["messages"][1]["role"] == "assistant"
        # response content matches the task's response
        if ex["messages"][0]["content"] == "task A":
            assert ex["messages"][1]["content"] == "resp A"
        else:
            assert ex["messages"][1]["content"] == "resp B"


def test_build_examples_dedups_repeated_task():
    c = _conn()
    ms.log_interaction(c, "a", "same task", "", "resp 1", "code")
    ms.log_interaction(c, "b", "same task", "", "resp 2", "code")
    ms.record_outcome_row(c, "a", "tests_passed", 1.0)
    ms.record_outcome_row(c, "b", "tests_passed", 1.0)

    examples = etd.build_examples(c)

    assert len(examples) == 1
