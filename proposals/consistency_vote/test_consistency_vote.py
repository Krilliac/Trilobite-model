import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import consistency_vote  # noqa


def test_normalize_strips_lowers_and_collapses_whitespace():
    assert consistency_vote.normalize("  Hello   World  ") == "hello world"
    assert consistency_vote.normalize("Hello\n\tWorld") == "hello world"
    assert consistency_vote.normalize("HELLO") == "hello"


def test_normalize_handles_none_and_empty():
    assert consistency_vote.normalize(None) == ""
    assert consistency_vote.normalize("") == ""
    assert consistency_vote.normalize("   ") == ""


def test_clear_majority_wins():
    v = consistency_vote.vote(["cat", "dog", "cat", "cat", "dog"])
    assert v.winner == "cat"
    assert v.representative == "cat"
    assert v.count == 3
    assert v.total == 5
    assert v.is_tie is False
    assert v.tied_with == ["cat"]
    assert v.has_majority is True  # 3/5 > half
    assert v.counts == {"cat": 3, "dog": 2}


def test_normalization_merges_formatting_variants():
    v = consistency_vote.vote(["  Cat ", "cat", "CAT   "])
    assert v.winner == "cat"
    assert v.count == 3
    assert v.total == 3
    assert v.has_majority is True


def test_representative_preserves_first_original_casing():
    v = consistency_vote.vote(["  Cat  ", "cat", "Dog"])
    # winner is "cat" (2 votes); representative is the FIRST original string
    # that normalized to the winner, not a normalized/lowercased copy.
    assert v.winner == "cat"
    assert v.representative == "  Cat  "


def test_plurality_without_majority():
    # "a" has 2 of 5 votes: it wins (plurality) but does not clear 50%.
    v = consistency_vote.vote(["a", "a", "b", "c", "d"])
    assert v.winner == "a"
    assert v.count == 2
    assert v.has_majority is False
    assert v.is_tie is False


def test_tie_rule_breaks_by_first_seen_order():
    # "b" appears first among the tied candidates -> "b" wins the tie,
    # even though "a" is alphabetically first and appears overall-first
    # only if we look at raw order; here "b" is genuinely first-seen.
    v = consistency_vote.vote(["b", "a", "a", "b"])
    assert v.is_tie is True
    assert v.count == 2
    assert v.winner == "b"
    assert set(v.tied_with) == {"a", "b"}


def test_tie_rule_first_candidate_wins_two_way_tie():
    v = consistency_vote.vote(["yes", "no"])
    assert v.is_tie is True
    assert v.winner == "yes"
    assert v.representative == "yes"
    assert v.has_majority is False  # 1/2 is not > half


def test_three_way_tie_all_reported():
    v = consistency_vote.vote(["x", "y", "z"])
    assert v.is_tie is True
    assert v.count == 1
    assert set(v.tied_with) == {"x", "y", "z"}
    assert v.winner == "x"  # first-seen


def test_single_candidate_is_a_trivial_majority():
    v = consistency_vote.vote(["only answer"])
    assert v.winner == "only answer"
    assert v.count == 1
    assert v.total == 1
    assert v.is_tie is False
    assert v.has_majority is True


def test_empty_candidates_raises():
    with pytest.raises(consistency_vote.EmptyCandidatesError):
        consistency_vote.vote([])


def test_empty_candidates_is_a_value_error():
    # EmptyCandidatesError should be catchable generically as ValueError.
    with pytest.raises(ValueError):
        consistency_vote.vote([])


def test_majority_answer_convenience_wrapper():
    candidates = ["Paris", "paris", "PARIS ", "London"]
    assert consistency_vote.majority_answer(candidates) == "Paris"


def test_majority_answer_propagates_empty_error():
    with pytest.raises(consistency_vote.EmptyCandidatesError):
        consistency_vote.majority_answer([])
