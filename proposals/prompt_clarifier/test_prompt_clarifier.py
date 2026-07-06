import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import prompt_clarifier as pc

WELL_SPECIFIED = """
Write a function named `count_vowels(s: str) -> int` that returns the number
of vowels (a, e, i, o, u, case-insensitive) in the string s.

Example: count_vowels("Hello World") -> 3
"""

VAGUE_PROMPT = "Write something that processes the data and returns the result."

MISSING_EXAMPLE_ONLY = (
    "Write a function named `add(a: int, b: int) -> int` that returns the "
    "sum of a and b."
)

MISSING_SIGNATURE_ONLY = (
    "Write code that takes two integers a and b and returns their sum. "
    "For example, given 3 and 4, it returns 7."
)

MISSING_TYPES_ONLY = (
    "Write a function called `transform` that takes some data and some "
    "items and returns the result. For example, transform(1, 2) -> 3."
)


def test_well_specified_prompt_yields_no_questions():
    assert pc.clarify(WELL_SPECIFIED) == []
    assert pc.is_well_specified(WELL_SPECIFIED) is True


def test_vague_prompt_yields_multiple_questions():
    questions = pc.clarify(VAGUE_PROMPT)
    assert len(questions) >= 2
    assert pc.is_well_specified(VAGUE_PROMPT) is False


def test_empty_prompt_yields_a_single_question():
    for empty in ("", "   ", None):
        questions = pc.clarify(empty)
        assert len(questions) == 1
        assert "empty" in questions[0].lower()


def test_clarify_returns_list_of_str():
    for text in (WELL_SPECIFIED, VAGUE_PROMPT, MISSING_EXAMPLE_ONLY, ""):
        result = pc.clarify(text)
        assert isinstance(result, list)
        assert all(isinstance(q, str) for q in result)


# --- signal 1: I/O example -----------------------------------------------------

def test_has_io_example_detects_doctest_style():
    assert pc.has_io_example("Do the thing.\n>>> foo(1)\n2\n")


def test_has_io_example_detects_eg():
    assert pc.has_io_example("Reverse a string, e.g. 'abc' -> 'cba'.")


def test_has_io_example_detects_for_example():
    assert pc.has_io_example("Sum a list. For example, [1, 2, 3] sums to 6.")


def test_has_io_example_detects_example_colon():
    assert pc.has_io_example("Do the thing.\nExample:\nfoo(1) -> 2")


def test_has_io_example_detects_input_output_block():
    assert pc.has_io_example("Do the thing.\nInput: [1, 2]\nOutput: 3")


def test_has_io_example_detects_literal_call_with_arrow():
    assert pc.has_io_example("Write is_even(n). is_even(4) -> True.")
    assert pc.has_io_example('is_palindrome("racecar") returns True')


def test_has_io_example_false_when_absent():
    assert not pc.has_io_example("Write a function that reverses a string.")
    assert not pc.has_io_example("")
    assert not pc.has_io_example(None)


def test_has_io_example_type_hint_alone_is_not_an_example():
    # a bare type-hinted signature is not itself a worked example
    assert not pc.has_io_example("def add(a: int, b: int) -> int: ...")


# --- signal 2: function/method signature ---------------------------------------

def test_has_function_signature_detects_def():
    assert pc.has_function_signature("def solve(x): return x")


def test_has_function_signature_detects_backtick_code():
    assert pc.has_function_signature("Implement `solve(x, y)` that adds them.")


def test_has_function_signature_detects_named():
    assert pc.has_function_signature("Write a function named solve that adds two numbers.")
    assert pc.has_function_signature("Write a method called Solve that adds two numbers.")


def test_has_function_signature_detects_function_paren_form():
    assert pc.has_function_signature("Write a function solve(a, b) that adds them.")


def test_has_function_signature_detects_class():
    assert pc.has_function_signature("Implement class Stack with push and pop.")


def test_has_function_signature_false_when_absent():
    assert not pc.has_function_signature("Write something that adds two numbers.")
    assert not pc.has_function_signature("")
    assert not pc.has_function_signature(None)


def test_has_function_signature_does_not_false_positive_on_plain_parenthetical():
    # a parenthetical aside should not read as a function call/signature
    text = "Handle the edge case (and log it) before returning."
    assert not pc.has_function_signature(text)


# --- signal 3: ambiguous types --------------------------------------------------

def test_ambiguous_type_terms_flags_vague_nouns():
    hits = pc.ambiguous_type_terms("It takes some data and returns the result.")
    assert "data" in hits
    assert "result" in hits


def test_ambiguous_type_terms_not_flagged_when_qualified_nearby():
    hits = pc.ambiguous_type_terms("It takes a list of integers and returns an int.")
    assert hits == []


def test_has_type_hints_detects_arrow_and_colon_hints():
    assert pc.has_type_hints("def add(a: int, b: int) -> int: ...")
    assert pc.has_type_hints("Parameter x: List[int]")
    assert pc.has_type_hints("returns Optional[str]")


def test_has_type_hints_false_when_absent():
    assert not pc.has_type_hints("Write a function that adds two numbers.")
    assert not pc.has_type_hints("")
    assert not pc.has_type_hints(None)


def test_clarify_skips_type_question_when_type_hints_present_even_with_vague_words():
    # "data" is vague, but the prompt already type-hints the signature, so
    # clarify() should not raise the ambiguous-types question.
    text = ("def transform(data: list[int]) -> int: "
            "'''sum the data.''' For example, transform([1, 2]) -> 3")
    questions = pc.clarify(text)
    assert not any("exact types" in q for q in questions)


# --- clarify() integration of the three signals --------------------------------

def test_clarify_flags_only_missing_example():
    questions = pc.clarify(MISSING_EXAMPLE_ONLY)
    assert len(questions) == 1
    assert "example" in questions[0].lower() or "input" in questions[0].lower()


def test_clarify_flags_only_missing_signature():
    questions = pc.clarify(MISSING_SIGNATURE_ONLY)
    assert len(questions) == 1
    assert "named" in questions[0].lower() or "parameters" in questions[0].lower()


def test_clarify_flags_only_ambiguous_types():
    questions = pc.clarify(MISSING_TYPES_ONLY)
    assert len(questions) == 1
    assert "types" in questions[0].lower()


def test_clarify_caps_number_of_vague_terms_shown():
    text = ("Write a function named process that takes some data, some items, "
            "some values, some things, some stuff, some params, and some args, "
            "then returns the result.")
    questions = pc.clarify(text)
    type_questions = [q for q in questions if "exact types" in q]
    assert len(type_questions) == 1
    shown = type_questions[0].split(":", 1)[1]
    # capped at _MAX_VAGUE_TERMS_SHOWN terms, comma-separated before the "?"
    terms_listed = shown.split("?")[0].split(",")
    assert len(terms_listed) <= pc._MAX_VAGUE_TERMS_SHOWN


def test_clarify_is_deterministic():
    assert pc.clarify(VAGUE_PROMPT) == pc.clarify(VAGUE_PROMPT)
    assert pc.clarify(WELL_SPECIFIED) == pc.clarify(WELL_SPECIFIED)


def test_is_well_specified_matches_clarify_emptiness():
    for text in (WELL_SPECIFIED, VAGUE_PROMPT, MISSING_EXAMPLE_ONLY,
                 MISSING_SIGNATURE_ONLY, MISSING_TYPES_ONLY, ""):
        assert pc.is_well_specified(text) == (pc.clarify(text) == [])
