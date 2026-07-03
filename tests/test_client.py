import json

import trilobite_client as tc


def test_build_request_with_key():
    url, headers, body = tc.build_request("http://h:1", "k", "hi")
    assert url == "http://h:1/v1/chat/completions"
    assert headers["Authorization"] == "Bearer k"
    assert headers["Content-Type"] == "application/json"
    obj = json.loads(body.decode("utf-8"))
    assert obj["model"] == "trilobite"
    assert obj["messages"] == [{"role": "user", "content": "hi"}]
    assert obj["stream"] is False


def test_build_request_no_key():
    url, headers, body = tc.build_request("http://h:1", "", "hi")
    assert "Authorization" not in headers
    obj = json.loads(body.decode("utf-8"))
    assert obj["messages"][0]["content"] == "hi"


def test_build_request_strips_trailing_slash():
    url, _, _ = tc.build_request("http://h:1/", "", "hi")
    assert url == "http://h:1/v1/chat/completions"
