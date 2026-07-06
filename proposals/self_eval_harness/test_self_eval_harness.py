import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import self_eval_harness as seh


TASKS = [{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}]


def _stub_solve(pass_names):
    """Stub solve_fn: passes iff the task's name is in pass_names. No model,
    no GPU, no network -- fully deterministic."""
    def solve(task):
        return task["name"] in pass_names
    return solve


# ---- run_eval ----

def test_run_eval_counts_pass_and_total():
    summary = seh.run_eval(TASKS, _stub_solve({"a", "c"}))
    assert summary["passed"] == 2
    assert summary["total"] == 4
    assert summary["pass_rate"] == 0.5
    assert [r["passed"] for r in summary["results"]] == [True, False, True, False]


def test_run_eval_empty_tasks_gives_zero_rate_not_crash():
    summary = seh.run_eval([], _stub_solve(set()))
    assert summary == {"results": [], "passed": 0, "total": 0, "pass_rate": 0.0}


def test_run_eval_treats_solve_fn_exception_as_fail():
    def flaky(task):
        if task["name"] == "b":
            raise RuntimeError("boom")
        return True
    summary = seh.run_eval(TASKS, flaky)
    assert summary["passed"] == 3  # a, c, d pass; b raised -> counted as fail
    assert summary["results"][1] == {"name": "b", "passed": False}


def test_run_eval_falls_back_to_index_name_for_non_dict_tasks():
    summary = seh.run_eval(["x", "y"], lambda t: t == "x")
    assert summary["results"][0]["name"] == "0"
    assert summary["results"][1]["name"] == "1"


# ---- append_log / load_log ----

def test_append_log_writes_record_with_injected_ts(tmp_path):
    log_path = str(tmp_path / "eval_log.jsonl")
    record = seh.append_log(log_path, ts=1000.0, passed=3, total=4)
    assert record == {"ts": 1000.0, "passed": 3, "total": 4, "pass_rate": 0.75}

    with open(log_path, encoding="utf-8") as f:
        line = f.readline()
    assert json.loads(line) == record


def test_append_log_creates_missing_parent_dir(tmp_path):
    log_path = str(tmp_path / "nested" / "dir" / "eval_log.jsonl")
    seh.append_log(log_path, ts=1.0, passed=1, total=1)
    assert os.path.exists(log_path)


def test_append_log_zero_total_gives_zero_rate_not_zerodiv(tmp_path):
    log_path = str(tmp_path / "eval_log.jsonl")
    record = seh.append_log(log_path, ts=5.0, passed=0, total=0)
    assert record["pass_rate"] == 0.0


def test_load_log_missing_file_returns_empty_list(tmp_path):
    assert seh.load_log(str(tmp_path / "nope.jsonl")) == []


def test_load_log_reads_back_multiple_records_in_order(tmp_path):
    log_path = str(tmp_path / "eval_log.jsonl")
    seh.append_log(log_path, ts=1.0, passed=1, total=2)
    seh.append_log(log_path, ts=2.0, passed=2, total=2)
    history = seh.load_log(log_path)
    assert [r["ts"] for r in history] == [1.0, 2.0]
    assert [r["pass_rate"] for r in history] == [0.5, 1.0]


def test_load_log_skips_blank_lines(tmp_path):
    log_path = str(tmp_path / "eval_log.jsonl")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"ts": 1.0, "passed": 1, "total": 1, "pass_rate": 1.0}) + "\n")
        f.write("\n")
        f.write(json.dumps({"ts": 2.0, "passed": 0, "total": 1, "pass_rate": 0.0}) + "\n")
    assert len(seh.load_log(log_path)) == 2


# ---- run_and_log (end-to-end with a stub solve_fn) ----

def test_run_and_log_end_to_end(tmp_path):
    log_path = str(tmp_path / "eval_log.jsonl")
    record = seh.run_and_log(TASKS, _stub_solve({"a", "b", "c"}), ts=42.0, log_path=log_path)

    assert record["ts"] == 42.0
    assert record["passed"] == 3
    assert record["total"] == 4
    assert record["pass_rate"] == 0.75
    assert len(record["results"]) == 4  # per-task detail attached for inspection

    # what actually landed on disk should NOT carry the per-task detail --
    # the log stays a compact regression-tracking summary.
    on_disk = seh.load_log(log_path)
    assert len(on_disk) == 1
    assert "results" not in on_disk[0]
    assert on_disk[0]["passed"] == 3


def test_run_and_log_accumulates_across_multiple_runs(tmp_path):
    log_path = str(tmp_path / "eval_log.jsonl")
    seh.run_and_log(TASKS, _stub_solve({"a", "b", "c", "d"}), ts=1.0, log_path=log_path)
    seh.run_and_log(TASKS, _stub_solve({"a"}), ts=2.0, log_path=log_path)
    history = seh.load_log(log_path)
    assert [r["pass_rate"] for r in history] == [1.0, 0.25]


# ---- regressed ----

def test_regressed_true_when_latest_drops_below_prior_best():
    history = [
        {"ts": 1.0, "pass_rate": 0.9},
        {"ts": 2.0, "pass_rate": 0.95},
        {"ts": 3.0, "pass_rate": 0.6},
    ]
    assert seh.regressed(history) is True


def test_regressed_false_when_latest_matches_or_beats_prior_best():
    history = [
        {"ts": 1.0, "pass_rate": 0.5},
        {"ts": 2.0, "pass_rate": 0.8},
        {"ts": 3.0, "pass_rate": 0.8},
    ]
    assert seh.regressed(history) is False


def test_regressed_false_with_fewer_than_two_records():
    assert seh.regressed([]) is False
    assert seh.regressed([{"ts": 1.0, "pass_rate": 0.1}]) is False


def test_regressed_respects_tolerance_for_small_noise():
    history = [{"ts": 1.0, "pass_rate": 0.80}, {"ts": 2.0, "pass_rate": 0.78}]
    assert seh.regressed(history, tolerance=0.0) is True
    assert seh.regressed(history, tolerance=0.05) is False


def test_regressed_with_explicit_latest_argument(tmp_path):
    # models a caller checking a fresh run against on-disk history without
    # re-reading the file it just appended to.
    history = [{"ts": 1.0, "pass_rate": 0.9}, {"ts": 2.0, "pass_rate": 0.85}]
    fresh = {"ts": 3.0, "pass_rate": 0.5}
    assert seh.regressed(history, latest=fresh) is True
    assert seh.regressed(history, latest={"ts": 3.0, "pass_rate": 0.95}) is False
