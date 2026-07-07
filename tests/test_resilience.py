"""Quota-resilience tests: 429 backoff, model fallback, clean exhaustion.

These test the retry/fallback logic in GeminiClient._generate by stubbing the
underlying SDK client — no network, no real key.
"""
from types import SimpleNamespace

import pytest

import researchpilot.llm as llm_mod
from researchpilot.llm import GeminiClient, QuotaExhausted


class FakeQuotaError(Exception):
    def __init__(self):
        super().__init__("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, "
                         "'message': 'You exceeded your current quota'}}")


def make_client(monkeypatch, script):
    """Build a GeminiClient whose SDK layer replays `script`.

    script items: exceptions (raised) or responses (returned).
    """
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)  # instant tests
    client = GeminiClient.__new__(GeminiClient)  # skip __init__ (no SDK needed)
    client._types = SimpleNamespace(
        GenerateContentConfig=lambda **kw: kw,
        Tool=lambda **kw: kw,
        GoogleSearch=lambda: None,
    )
    client.max_api_calls = 25
    client.calls_made = 0
    client.pacing = 0
    client._last_call = None
    client.model = "model-a"
    client._fallback = ["model-b", "model-c"]
    calls = []

    def fake_generate_content(model, **kwargs):
        calls.append(model)
        item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=fake_generate_content))
    return client, calls


def test_backoff_then_success_on_transient_429(monkeypatch):
    ok = SimpleNamespace(text="fine")
    client, calls = make_client(monkeypatch, [FakeQuotaError(), ok])
    assert client.generate_text("p") == "fine"
    assert calls == ["model-a", "model-a"]  # retried same model once


def test_falls_back_to_next_model_after_two_429s(monkeypatch):
    ok = SimpleNamespace(text="fine")
    client, calls = make_client(monkeypatch,
                                [FakeQuotaError(), FakeQuotaError(), ok])
    assert client.generate_text("p") == "fine"
    assert calls == ["model-a", "model-a", "model-b"]
    assert client.model == "model-b"          # sticks for subsequent calls
    assert client._fallback == ["model-c"]


def test_quota_exhausted_when_all_models_fail(monkeypatch):
    client, calls = make_client(monkeypatch, [FakeQuotaError()] * 8)
    with pytest.raises(QuotaExhausted):
        client.generate_text("p")
    # tried model-a twice, model-b twice, model-c twice -> gave up
    assert calls == ["model-a", "model-a", "model-b", "model-b",
                     "model-c", "model-c"]


def test_non_quota_errors_are_not_swallowed(monkeypatch):
    client, _ = make_client(monkeypatch, [RuntimeError("boom")])
    with pytest.raises(RuntimeError, match="boom"):
        client.generate_text("p")


def test_is_quota_error_detection():
    assert GeminiClient._is_quota_error(FakeQuotaError())
    assert not GeminiClient._is_quota_error(ValueError("bad json"))
    assert GeminiClient._is_quota_error(
        Exception("google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED"))
