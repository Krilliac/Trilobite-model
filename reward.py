"""Outcome signal -> scalar reward. Execution-grounded signals weighted highest."""

SIGNAL_REWARDS = {
    "tests_passed": 1.0,
    "accepted": 0.8,
    "compiled": 0.7,
    "rejected": -0.5,
    "failed": -1.0,
}
VALID_SIGNALS = set(SIGNAL_REWARDS)
GOOD_THRESHOLD = 0.7


def score(signal):
    return SIGNAL_REWARDS.get(signal, 0.0)


def is_good(signal):
    return score(signal) >= GOOD_THRESHOLD
