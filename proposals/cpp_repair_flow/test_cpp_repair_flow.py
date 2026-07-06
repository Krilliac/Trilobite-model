"""Tests for cpp_repair_flow.solve_cpp — all deterministic, no GPU/network/compiler.

gen_fn and verify_fn are stubbed throughout: gen_fn is a canned-response queue
(no model call), verify_fn is a pure-python "fake compiler" that fails while a
marker string is present in the candidate and passes once it's gone. This
proves the generate->verify->repair loop actually converges broken->fixed
(and gives up cleanly when it can't), without touching MSVC/vcvars.
"""
import verifiers  # for Verdict + (in one test) monkeypatching verifiers.verify

import cpp_repair_flow as crf

BUG_MARKER = "int x = 1 int y = 2;"  # stands in for a real syntax error
FIXED_LINE = "int x = 1; int y = 2;"


def marker_verify_fn(code):
    """Fake compiler: fails with a canned MSVC-shaped error while BUG_MARKER is
    present, passes once the caller has fixed it. Mirrors verifiers.cpp_compile's
    Verdict(passed, reason, detail) contract without invoking cl.exe."""
    if BUG_MARKER in (code or ""):
        return verifiers.Verdict(False, "error C2143", "tu.cpp(2): error C2143: missing ';' before 'int'")
    return verifiers.Verdict(True, "compiled", "compiled")


def canned_gen_fn(responses):
    """Returns a gen_fn(prompt) that ignores its argument and pops the next
    canned reply off `responses` each call (recording every prompt it saw)."""
    calls = []
    it = iter(responses)

    def gen_fn(prompt):
        calls.append(prompt)
        return next(it)
    gen_fn.calls = calls
    return gen_fn


def cpp_block(body):
    return "```\n%s\n```" % body


# --- happy path: converges broken -> fixed ----------------------------------
def test_solve_cpp_converges_broken_to_fixed():
    broken = cpp_block("int main(){ %s return 0; }" % BUG_MARKER)
    fixed = cpp_block("int main(){ %s return 0; }" % FIXED_LINE)
    gen_fn = canned_gen_fn([broken, fixed])

    result = crf.solve_cpp("write a function that adds two ints", gen_fn,
                           max_attempts=3, verify_fn=marker_verify_fn)

    assert result["passed"] is True
    assert result["attempts"] == 2
    assert BUG_MARKER not in result["code"]
    assert FIXED_LINE in result["code"]
    assert len(result["transcript"]) == 2
    assert result["transcript"][0]["ok"] is False
    assert result["transcript"][1]["ok"] is True
    # the repair prompt fed to attempt 2 must carry the compiler diagnostic
    # forward, and the original failing code, so the model can see what broke.
    assert len(gen_fn.calls) == 2
    assert "error C2143" in gen_fn.calls[1]
    assert BUG_MARKER in gen_fn.calls[1]


# --- gives up cleanly when the generator never fixes the bug ----------------
def test_solve_cpp_gives_up_after_max_attempts():
    broken = cpp_block("int main(){ %s return 0; }" % BUG_MARKER)
    gen_fn = canned_gen_fn([broken, broken, broken])

    result = crf.solve_cpp("write a function", gen_fn, max_attempts=3,
                           verify_fn=marker_verify_fn)

    assert result["passed"] is False
    assert result["attempts"] == 3
    assert len(result["transcript"]) == 3
    assert all(entry["ok"] is False for entry in result["transcript"])
    assert BUG_MARKER in result["code"]


# --- a reply with no fenced code block is handled, not a crash --------------
def test_solve_cpp_recovers_from_missing_code_block():
    no_block = "Sure, here is the fix (I forgot to fence it)."
    fixed = cpp_block("int main(){ %s return 0; }" % FIXED_LINE)
    gen_fn = canned_gen_fn([no_block, fixed])

    result = crf.solve_cpp("write a function", gen_fn, max_attempts=3,
                           verify_fn=marker_verify_fn)

    assert result["passed"] is True
    assert result["attempts"] == 2
    assert result["transcript"][0]["code"] is None
    assert result["transcript"][0]["ok"] is False
    assert result["transcript"][1]["ok"] is True


# --- default wiring reaches verifiers.verify('cpp_compile', ...) ------------
def test_solve_cpp_default_wiring_dispatches_through_registry(monkeypatch):
    """With no verify_fn override, solve_cpp must still route through the
    'cpp_compile' registry key (with the given spec) rather than some other
    verifier — proven by monkeypatching verifiers.verify itself so no real
    compiler is ever invoked."""
    seen = {}

    def fake_verify(name, artifact, spec=None):
        seen["name"] = name
        seen["artifact"] = artifact
        seen["spec"] = spec
        return verifiers.Verdict(True, "compiled", "compiled")

    monkeypatch.setattr(verifiers, "verify", fake_verify)

    ok_code = cpp_block("int main(){ return 0; }")
    gen_fn = canned_gen_fn([ok_code])
    spec = {"std": "c++20"}

    result = crf.solve_cpp("write a no-op main", gen_fn, max_attempts=1, spec=spec)

    assert result["passed"] is True
    assert seen["name"] == "cpp_compile"
    assert seen["spec"] == spec
    assert "int main" in seen["artifact"]


# --- an injected verify_fn takes priority over spec/registry ----------------
def test_solve_cpp_injected_verify_fn_bypasses_registry(monkeypatch):
    def exploding_verify(name, artifact, spec=None):
        raise AssertionError("verifiers.verify should not be called when verify_fn is given")

    monkeypatch.setattr(verifiers, "verify", exploding_verify)

    ok_code = cpp_block("int main(){ return 0; }")
    gen_fn = canned_gen_fn([ok_code])

    result = crf.solve_cpp("write a no-op main", gen_fn, max_attempts=1,
                           spec={"std": "c++20"}, verify_fn=marker_verify_fn)

    assert result["passed"] is True
