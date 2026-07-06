import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from fewshot_injector import build_fewshot_prompt, _extract, _truncate


def test_no_recalls_returns_task_unchanged():
    assert build_fewshot_prompt("write a fib function", []) == "write a fib function"
    assert build_fewshot_prompt("write a fib function", None) == "write a fib function"


def test_k_zero_or_negative_returns_task_unchanged():
    recalls = ["fib -> def fib(n): ..."]
    assert build_fewshot_prompt("task", recalls, k=0) == "task"
    assert build_fewshot_prompt("task", recalls, k=-1) == "task"


def test_string_recalls_from_recall_module_format():
    recalls = [
        "reverse a string -> def rev(s): return s[::-1]",
        "check palindrome -> def is_pal(s): return s == s[::-1]",
    ]
    out = build_fewshot_prompt("title-case a string", recalls, k=2)
    assert "Task: reverse a string" in out
    assert "def rev(s): return s[::-1]" in out
    assert "Task: check palindrome" in out
    assert "# Now solve this new task:" in out
    assert out.strip().endswith("title-case a string")
    assert "Example 1" in out and "Example 2" in out


def test_string_recall_with_no_arrow_has_empty_solution():
    task, solution, score = _extract("just a bare task string")
    assert task == "just a bare task string"
    assert solution == ""
    assert score is None


def test_dict_recalls_various_key_names():
    recalls = [
        {"task": "sort a list", "solution": "sorted(xs)"},
        {"query": "dedupe a list", "response": "list(set(xs))"},
        {"prompt": "sum a list", "answer": "sum(xs)"},
        {"task": "flatten nested", "code": "[y for x in xs for y in x]"},
    ]
    out = build_fewshot_prompt("new task", recalls, k=4)
    assert "sort a list" in out and "sorted(xs)" in out
    assert "dedupe a list" in out and "list(set(xs))" in out
    assert "sum a list" in out and "sum(xs)" in out
    assert "flatten nested" in out and "[y for x in xs for y in x]" in out


def test_tuple_and_list_recalls():
    recalls = [("task A", "solution A"), ["task B", "solution B"]]
    out = build_fewshot_prompt("t", recalls, k=2)
    assert "task A" in out and "solution A" in out
    assert "task B" in out and "solution B" in out


def test_k_limits_examples_to_top_k():
    recalls = [("t%d" % i, "s%d" % i) for i in range(10)]
    out = build_fewshot_prompt("new", recalls, k=3)
    assert out.count("--- Example") == 3
    assert "t0" in out and "t1" in out and "t2" in out
    assert "t3" not in out


def test_scored_recalls_are_sorted_descending_by_similarity():
    recalls = [
        {"task": "low", "solution": "s_low", "score": 0.1},
        {"task": "high", "solution": "s_high", "score": 0.9},
        {"task": "mid", "solution": "s_mid", "similarity": 0.5},
    ]
    out = build_fewshot_prompt("new", recalls, k=3)
    assert out.index("high") < out.index("mid") < out.index("low")


def test_mixed_scored_and_unscored_preserves_given_order():
    # Not every entry has a score -> sorting is skipped, original order kept.
    recalls = [
        {"task": "first", "solution": "s1", "score": 0.9},
        {"task": "second", "solution": "s2"},  # no score
    ]
    out = build_fewshot_prompt("new", recalls, k=2)
    assert out.index("first") < out.index("second")


def test_long_solution_is_truncated():
    long_solution = "x" * 1000
    recalls = [("task", long_solution)]
    out = build_fewshot_prompt("new", recalls, k=1, max_solution_chars=50)
    assert "x" * 51 not in out
    assert "..." in out


def test_truncate_disabled_when_max_chars_falsy():
    assert _truncate("hello world", 0) == "hello world"
    assert _truncate("hello world", None) == "hello world"


def test_unknown_entry_shape_falls_back_to_stringify_not_raise():
    # int has no recognizable task/solution fields; _extract stringifies it as the
    # task text rather than raising. A dict with no recognized keys yields empty
    # task+solution and is dropped entirely.
    recalls = [12345, {"unrelated_key": "value"}]
    out = build_fewshot_prompt("new", recalls, k=2)
    assert out.count("--- Example") == 1
    assert "Task: 12345" in out
    assert "unrelated_key" not in out


def test_partial_unknown_entry_mixed_with_valid_one():
    recalls = [{"unrelated_key": "value"}, ("real task", "real solution")]
    out = build_fewshot_prompt("new", recalls, k=2)
    assert "real task" in out
    assert "real solution" in out
    assert out.count("--- Example") == 1


def test_entry_with_only_task_no_solution_gets_placeholder():
    recalls = [("lonely task", "")]
    out = build_fewshot_prompt("new", recalls, k=1)
    assert "lonely task" in out
    assert "(no solution recorded)" in out


def test_pure_no_side_effects_deterministic():
    recalls = [("a", "b"), ("c", "d")]
    out1 = build_fewshot_prompt("task", recalls, k=2)
    out2 = build_fewshot_prompt("task", recalls, k=2)
    assert out1 == out2


def test_task_none_treated_as_empty_string():
    assert build_fewshot_prompt(None, []) == ""
