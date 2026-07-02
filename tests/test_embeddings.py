import embeddings as e


class FakeResponse:
    """Fake response object for mocking urllib.request.urlopen."""
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_blob_roundtrip():
    v = [0.5, -1.25, 3.0]
    back = e.from_blob(e.to_blob(v))
    assert len(back) == 3
    assert abs(back[0] - 0.5) < 1e-6
    assert abs(back[1] + 1.25) < 1e-6


def test_cosine_identical_is_one():
    assert abs(e.cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6


def test_cosine_orthogonal_is_zero():
    assert abs(e.cosine([1.0, 0.0], [0.0, 1.0])) < 1e-6


def test_cosine_handles_empty():
    assert e.cosine([], [1.0]) == 0.0
    assert e.cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_embed_soft_fails_to_none(monkeypatch):
    def boom(*a, **k):
        raise OSError("no ollama")

    monkeypatch.setattr(e.urllib.request, "urlopen", boom)
    assert e.embed("anything") is None


def test_embed_success_path(monkeypatch):
    def mock_urlopen(*a, **k):
        return FakeResponse(b'{"embedding": [0.1, 0.2, 0.3]}')

    monkeypatch.setattr(e.urllib.request, "urlopen", mock_urlopen)
    result = e.embed("hi")
    assert result is not None
    assert len(result) == 3
    assert abs(result[0] - 0.1) < 1e-6
    assert abs(result[1] - 0.2) < 1e-6
    assert abs(result[2] - 0.3) < 1e-6


def test_embed_non_dict_json_returns_none(monkeypatch):
    def mock_urlopen(*a, **k):
        return FakeResponse(b'[1, 2, 3]')

    monkeypatch.setattr(e.urllib.request, "urlopen", mock_urlopen)
    result = e.embed("hi")
    assert result is None
