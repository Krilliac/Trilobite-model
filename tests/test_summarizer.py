import summarizer


def test_summarize_includes_turns_and_prior_summary():
    seen = {}

    def fake(prompt, tier, system, temperature, num_predict):
        seen["prompt"] = prompt
        seen["tier"] = tier
        return "  updated summary  "

    out = summarizer.summarize("old summary", [("q1", "a1"), ("q2", "a2")], fake)
    assert out == "updated summary"           # stripped
    assert seen["tier"] == "fast"             # cheap tier
    assert "old summary" in seen["prompt"]    # incremental: prior summary folded in
    assert "q1" in seen["prompt"] and "a2" in seen["prompt"]


def test_summarize_without_prior_summary():
    def fake(prompt, tier, system, temperature, num_predict):
        assert "PREVIOUS SUMMARY" not in prompt
        return "s"
    assert summarizer.summarize(None, [("q", "a")], fake) == "s"


def test_make_title_uses_model_output():
    def fake(prompt, tier, system, temperature, num_predict):
        return '"Fix the parser"\nextra'
    # strips quotes, takes first line
    assert summarizer.make_title("please fix the parser bug", fake) == "Fix the parser"


def test_make_title_falls_back_on_error():
    def boom(**kwargs):
        raise RuntimeError("down")
    title = summarizer.make_title("implement a red-black tree please", boom)
    assert title == "implement a red-black tree please"[:40]


def test_make_title_falls_back_on_empty():
    assert summarizer.make_title("hello world", lambda **k: "") == "hello world"
