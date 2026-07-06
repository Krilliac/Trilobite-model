import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import difficulty_estimator as de

EASY_TASK = "Write a function that adds two numbers and returns the sum."

HARD_TASK = """
Implement a thread-safe LRU cache in Python that supports concurrent access
from multiple threads without race conditions or deadlocks.

Requirements:
1. get(key) and put(key, value) must run in O(1) amortized time.
2. The cache must be safe under concurrent threads using a lock; no thread
   may observe a torn read.
3. Eviction must use a recursive helper to walk the internal doubly linked
   list at least once per put when the cache is full.
4. You must handle the edge case where capacity is zero.
5. The implementation should use dynamic programming style memoization for
   any repeated subcomputation of eviction cost, and must not exceed O(n)
   additional memory for n entries.
6. Parsing of a config string (e.g. "capacity=10;policy=lru") must be done
   with a regular expression, not manual string splitting.
"""

MEDIUM_TASK = (
    "Write a function that parses a simple arithmetic expression string "
    "(e.g. '3 + 4 * 2') and returns its integer value. It should handle "
    "at least the +, -, *, and / operators and respect operator precedence."
)


def test_score_is_in_unit_interval_for_various_inputs():
    for text in (EASY_TASK, HARD_TASK, MEDIUM_TASK, "", "x" * 5000):
        s = de.score(text)
        assert 0.0 <= s <= 1.0


def test_empty_task_scores_zero():
    assert de.score("") == 0.0
    assert de.score(None) == 0.0


def test_hard_task_scores_higher_than_easy_task():
    assert de.score(HARD_TASK) > de.score(EASY_TASK)


def test_hard_task_scores_higher_than_medium_task():
    assert de.score(HARD_TASK) > de.score(MEDIUM_TASK)


def test_medium_task_scores_higher_than_easy_task():
    assert de.score(MEDIUM_TASK) > de.score(EASY_TASK)


def test_keyword_detection_finds_expected_categories():
    result = de.estimate(HARD_TASK)
    assert "recursion" in result["keywords"]
    assert "dynamic_programming" in result["keywords"]
    assert "concurrency" in result["keywords"]
    assert "parsing" in result["keywords"]
    assert "regex" in result["keywords"]


def test_keyword_detection_finds_nothing_in_easy_task():
    result = de.estimate(EASY_TASK)
    assert result["keywords"] == []
    assert result["keyword_score"] == 0.0


def test_keyword_category_fires_once_regardless_of_repeat_mentions():
    once = "This uses recursion to solve the problem."
    many = "recursion recursion recursion recursion recursion recursion " * 5
    r_once = de.estimate(once)
    r_many = de.estimate(many)
    # both should report the same single category hit and the same
    # keyword_score contribution from it (length differs, keyword doesn't).
    assert r_once["keywords"] == ["recursion"]
    assert r_many["keywords"] == ["recursion"]
    assert r_once["keyword_score"] == r_many["keyword_score"]


def test_constraint_count_increases_with_more_requirements():
    low = de.estimate("Write a function that sorts a list.")
    high = de.estimate(HARD_TASK)
    assert high["constraint_count"] > low["constraint_count"]
    assert high["constraint_score"] > low["constraint_score"]


def test_constraint_patterns_detect_must_should_and_bounds():
    text = ("You must validate input. At most 10 items are allowed. "
            "It should run in O(n log n). 1. handle nulls 2. handle empties")
    result = de.estimate(text)
    assert result["constraint_count"] >= 4


def test_length_score_saturates_and_never_exceeds_one():
    short = de._length_score("hi")
    long_ = de._length_score("x" * 5000)
    assert 0.0 < short < long_ < 1.0


def test_length_score_is_zero_for_empty_string():
    assert de._length_score("") == 0.0


def test_estimate_returns_full_breakdown_keys():
    result = de.estimate(MEDIUM_TASK)
    for key in ("score", "length_score", "keyword_score", "constraint_score",
                "keywords", "constraint_count"):
        assert key in result


def test_score_matches_estimate_score_field():
    text = HARD_TASK
    assert de.score(text) == de.estimate(text)["score"]


def test_classify_buckets_are_monotonic_and_cover_range():
    assert de.classify(0.0) == "easy"
    assert de.classify(de.EASY_MAX) == "easy"
    assert de.classify(de.EASY_MAX + 0.01) == "medium"
    assert de.classify(de.MEDIUM_MAX) == "medium"
    assert de.classify(de.MEDIUM_MAX + 0.01) == "hard"
    assert de.classify(1.0) == "hard"


def test_classify_agrees_with_actual_scored_tasks():
    assert de.classify(de.score(EASY_TASK)) == "easy"
    assert de.classify(de.score(HARD_TASK)) == "hard"


def test_suggest_max_attempts_is_clamped_and_monotonic_in_difficulty():
    easy_attempts = de.suggest_max_attempts(EASY_TASK)
    hard_attempts = de.suggest_max_attempts(HARD_TASK)
    assert 1 <= easy_attempts <= 6
    assert 1 <= hard_attempts <= 6
    assert hard_attempts >= easy_attempts


def test_suggest_max_attempts_respects_custom_bounds():
    n = de.suggest_max_attempts(HARD_TASK, base=3, min_attempts=1, max_attempts=4)
    assert n <= 4
    n2 = de.suggest_max_attempts(EASY_TASK, base=3, min_attempts=2, max_attempts=6)
    assert n2 >= 2


def test_weights_override_is_renormalized_and_still_bounded():
    # a lopsided weights dict that doesn't sum to 1 should be renormalized,
    # not silently break the [0, 1] bound.
    result = de.estimate(HARD_TASK, weights={"length": 1, "keyword": 1, "constraint": 1})
    assert 0.0 <= result["score"] <= 1.0


def test_weights_override_can_zero_out_a_signal():
    result = de.estimate(HARD_TASK, weights={"length": 0, "keyword": 1, "constraint": 0})
    # with only the keyword signal active, score should equal its keyword_score
    assert math_close(result["score"], result["keyword_score"])


def math_close(a, b, eps=1e-9):
    return abs(a - b) < eps


def test_deterministic_repeat_calls_give_identical_result():
    r1 = de.estimate(HARD_TASK)
    r2 = de.estimate(HARD_TASK)
    assert r1 == r2
