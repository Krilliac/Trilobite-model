import trilobite_serve as ts


def test_history_from_messages_excludes_last_user_turn():
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "follow up"},  # current prompt -> excluded
    ]
    hist = ts._history_from_messages(messages)
    assert hist == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]


def test_history_from_messages_drops_system_and_empty():
    messages = [
        {"role": "system", "content": "you are trilobite"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": ""},   # empty -> dropped
        {"role": "user", "content": "current"},
    ]
    hist = ts._history_from_messages(messages)
    assert hist == [{"role": "user", "content": "q1"}]


def test_history_from_messages_single_turn_is_empty():
    hist = ts._history_from_messages([{"role": "user", "content": "only turn"}])
    assert hist == []


def test_history_from_messages_handles_empty():
    assert ts._history_from_messages([]) == []
    assert ts._history_from_messages(None) == []
