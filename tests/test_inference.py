from memorygraph.inference import ask_ollama


def test_ollama_token_counts_and_response_schema(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "Hello", "prompt_eval_count": 12, "eval_count": 3}

    def fake_post(url, json, timeout):
        assert url.startswith("http://127.0.0.1")
        assert json["options"]["temperature"] == 0
        assert json["options"]["num_predict"] == 128
        assert timeout == 120
        return Response()

    monkeypatch.setattr("memorygraph.inference.requests.post", fake_post)
    result = ask_ollama("Hi", "local-model")
    assert result.text == "Hello"
    assert result.prompt_tokens == 12
    assert result.output_tokens == 3
    assert result.elapsed_seconds >= 0
