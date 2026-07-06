"""consistency_vote — self-consistency majority vote over N candidate answers.

For NON-checkable tasks (open questions, explanations, free-text answers) there
is no verifier oracle like the ones in verifiers.py, so solver.solve_verified()
and best_of_n() can't tell a good candidate from a bad one by execution. The
self-consistency trick (Wang et al., 2022) sidesteps that: sample the model N
times at nonzero temperature, normalize each answer, and let the candidates
vote. Agreement across independently-sampled attempts is itself a (weak, but
free) correctness signal — it costs zero extra verifier infrastructure and
composes with anything that already produces a list of candidate strings
(best_of_n's transcript, a persona ensemble, a self-curriculum re-ask).

Pure, stdlib-only, deterministic: no model calls happen in this module — it
only tallies strings the caller already generated.
"""
import collections
import re

_WS_RE = re.compile(r"\s+")


class EmptyCandidatesError(ValueError):
    """Raised when vote()/majority_answer() is given zero candidates."""


Vote = collections.namedtuple(
    "Vote",
    ["winner", "representative", "count", "total", "is_tie", "tied_with", "has_majority", "counts"],
)


def normalize(text):
    """Strip, lowercase, and collapse all internal whitespace runs to one space.

    Collapses newlines/tabs too, so multi-line answers that differ only in
    formatting still compare equal.
    """
    return _WS_RE.sub(" ", (text or "").strip()).lower()


def vote(candidates):
    """Tally normalized `candidates` and return a Vote for the majority answer.

    Tie rule: ties are broken by FIRST-SEEN order among the tied candidates
    (the earliest-appearing normalized answer wins) — this relies on the
    documented CPython behavior that Counter.most_common() orders equal-count
    elements in the order first encountered, so the rule is deterministic and
    needs no extra bookkeeping here.

    `winner` and `tied_with` are normalized strings (for comparison); use
    `representative` to get back one of the caller's original, non-normalized
    candidate strings that produced the winner (the first one seen).

    Raises EmptyCandidatesError if `candidates` is empty.
    """
    if not candidates:
        raise EmptyCandidatesError("vote() requires at least one candidate")

    normalized = [normalize(c) for c in candidates]
    counts = collections.Counter(normalized)
    ranked = counts.most_common()  # ties ordered by first-seen (see docstring)

    top_count = ranked[0][1]
    tied_with = [text for text, cnt in ranked if cnt == top_count]
    winner = ranked[0][0]

    representative = next(
        orig for orig, norm in zip(candidates, normalized) if norm == winner
    )

    total = len(candidates)
    return Vote(
        winner=winner,
        representative=representative,
        count=top_count,
        total=total,
        is_tie=len(tied_with) > 1,
        tied_with=tied_with,
        has_majority=top_count > total / 2,
        counts=dict(counts),
    )


def majority_answer(candidates):
    """Convenience wrapper: just the winning candidate's original text.

    Equivalent to vote(candidates).representative — for callers that only
    want the answer string, not the vote breakdown.
    """
    return vote(candidates).representative
