"""Non-GPU tests for eval_retrieval.py: pool shape and true hold-out only.
Must never call the model -- no GPU/Ollama needed to run this file.
"""
import eval_retrieval
import training_tasks


def test_heldout_pool_has_at_least_ten_tasks():
    assert len(eval_retrieval.HELDOUT) >= 10


def test_heldout_is_a_true_holdout_disjoint_from_training_pool():
    training_names = {t["name"] for t in training_tasks.TASKS}
    heldout_names = {t["name"] for t in eval_retrieval.HELDOUT}
    overlap = heldout_names & training_names
    assert overlap == set(), "held-out task names must not appear in training_tasks.TASKS: %s" % overlap


def test_heldout_tasks_well_formed():
    names = set()
    for t in eval_retrieval.HELDOUT:
        assert t["name"].strip()
        assert t["prompt"].strip()
        assert t["check"].strip()
        names.add(t["name"])
    # also distinct amongst themselves
    assert len(names) == len(eval_retrieval.HELDOUT)
