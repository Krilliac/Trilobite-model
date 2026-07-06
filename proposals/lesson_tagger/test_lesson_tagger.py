import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import lesson_tagger as lt


# --- basic contract -----------------------------------------------------------

def test_tags_is_sorted_and_nonempty():
    assert list(lt.TAGS) == sorted(lt.TAGS)
    assert len(lt.TAGS) >= 9  # python, sql, cpp, javascript, concurrency,
                              # security, algorithm, io, testing, ...


def test_empty_and_none_input_returns_empty_list():
    assert lt.tag("") == []
    assert lt.tag("   \n\t  ") == []
    assert lt.tag(None) == []
    assert lt.tag_scores("") == {}
    assert lt.tag_scores(None) == {}


def test_result_is_sorted_list_of_known_tags():
    result = lt.tag("A lesson about Python and SQL together.")
    assert result == sorted(result)
    assert set(result).issubset(set(lt.TAGS))


# --- per-tag positive signals --------------------------------------------------

def test_python_signal():
    text = "def solve(self):\n    self.value = None\n    import os"
    assert "python" in lt.tag(text)


def test_sql_signal():
    text = "SELECT * FROM users WHERE active = 1 ORDER BY created_at"
    assert "sql" in lt.tag(text)


def test_cpp_signal():
    text = "Use std::unique_ptr<Widget> instead of a raw pointer; add nullptr checks."
    assert "cpp" in lt.tag(text)


def test_javascript_signal():
    text = "const total = items.reduce((a, b) => a + b, 0); console.log(total);"
    assert "javascript" in lt.tag(text)


def test_concurrency_signal():
    text = "The worker thread hit a deadlock because two locks were acquired out of order."
    assert "concurrency" in lt.tag(text)


def test_security_signal():
    text = "The endpoint was vulnerable to SQL injection; sanitize user input before querying."
    tags = lt.tag(text)
    assert "security" in tags
    assert "sql" in tags  # both signals legitimately fire on the same lesson


def test_algorithm_signal():
    text = "Switching to a binary search cut the lookup from O(n) to O(log n)."
    assert "algorithm" in lt.tag(text)


def test_io_signal():
    text = "Always use `with open(path, 'r', encoding='utf-8') as f:` to avoid leaking file handles."
    assert "io" in lt.tag(text)


def test_testing_signal():
    text = "Added a pytest fixture and an assertion to cover the regression."
    assert "testing" in lt.tag(text)


def test_git_signal():
    text = "Rebase the feature branch onto main before opening the pull request."
    assert "git" in lt.tag(text)


def test_networking_signal():
    text = "The REST endpoint timed out because the socket never received a TCP ack."
    assert "networking" in lt.tag(text)


def test_regex_signal():
    text = "Use re.sub with a regular expression instead of manual string splitting."
    assert "regex" in lt.tag(text)


def test_performance_signal():
    text = "Profiling showed the cache miss rate was hurting throughput; added memoization."
    assert "performance" in lt.tag(text)


def test_windows_signal():
    text = "On Windows, run the build through vcvars64.bat and check HKLM registry keys via PowerShell."
    tags = lt.tag(text)
    assert "windows" in tags


# --- multi-tag and negative cases -----------------------------------------------

def test_multiple_tags_fire_on_a_mixed_lesson():
    text = (
        "A Python asyncio worker used a lock around a shared dict; forgetting it "
        "caused a race condition under concurrent requests."
    )
    tags = lt.tag(text)
    assert "python" in tags
    assert "concurrency" in tags


def test_unrelated_text_yields_no_tags():
    text = "The weather was pleasant and the coffee was excellent this morning."
    assert lt.tag(text) == []


def test_cpp_text_does_not_get_javascript_tag():
    text = "Use std::unique_ptr<Widget> with nullptr checks and constexpr sizes."
    tags = lt.tag(text)
    assert "cpp" in tags
    assert "javascript" not in tags


# --- tag_scores ------------------------------------------------------------------

def test_tag_scores_counts_multiple_hits():
    text = "test test test pytest assert assert"
    scores = lt.tag_scores(text)
    assert scores["testing"] >= 4  # at least: test x3, pytest, assert x2 as separate patterns


def test_tag_scores_keys_are_subset_of_tag_result():
    text = "A deadlock occurred in the thread pool during the pytest run."
    scores = lt.tag_scores(text)
    tags = lt.tag(text)
    assert set(scores) == set(tags)


# --- filter_by_tag -----------------------------------------------------------------

def test_filter_by_tag_with_dicts():
    lessons = [
        {"id": 1, "text": "Always close SQL connections with a context manager."},
        {"id": 2, "text": "Use a mutex to guard the shared counter across threads."},
        {"id": 3, "text": "Prefer f-strings over % formatting in Python."},
    ]
    sql_only = lt.filter_by_tag(lessons, "sql")
    assert [item["id"] for item in sql_only] == [1]

    concurrency_only = lt.filter_by_tag(lessons, "concurrency")
    assert [item["id"] for item in concurrency_only] == [2]


def test_filter_by_tag_is_case_insensitive_for_the_wanted_tag():
    lessons = [{"text": "Rebase before opening a pull request on GitHub."}]
    assert lt.filter_by_tag(lessons, "GIT") == lessons


def test_filter_by_tag_custom_text_key():
    lessons = [{"body": "SELECT id FROM lessons WHERE tag = 'sql'"}]
    assert lt.filter_by_tag(lessons, "sql", text_key="body") == lessons
