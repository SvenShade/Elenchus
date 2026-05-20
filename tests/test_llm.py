from __future__ import annotations

import json
from types import SimpleNamespace

from llm_mcts.config import LLMConfig
from llm_mcts.llm import LLMClient
from llm_mcts.trace import TraceWriter


class StubJSONClient(LLMClient):
    def __init__(self, outputs, tracer):
        self.outputs = list(outputs)
        self.tracer = tracer
        self.chat_calls = []

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        return self.outputs.pop(0)


def test_chat_json_repairs_malformed_response(tmp_path):
    tracer = TraceWriter(run_dir=tmp_path / "run", enabled=True)
    client = StubJSONClient(["not json", '{"ok": true}'], tracer)

    parsed = client.chat_json(
        prompt_name="judge_rollout_state",
        messages=[{"role": "user", "content": "return json"}],
    )

    assert parsed == {"ok": True}


def test_chat_json_fallback_is_traced(tmp_path):
    tracer = TraceWriter(run_dir=tmp_path / "run", enabled=True)
    client = StubJSONClient(["nope", "still nope"], tracer)

    parsed = client.chat_json(
        prompt_name="policy_prior",
        messages=[{"role": "user", "content": "return json"}],
        fallback={"clarify": 1.0},
    )

    assert parsed == {"clarify": 1.0}
    lines = tracer.calls_path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    assert any(event["prompt_name"] == "policy_prior.fallback" for event in events)


def test_chat_json_passes_max_tokens_to_initial_and_repair_calls(tmp_path):
    tracer = TraceWriter(run_dir=tmp_path / "run", enabled=True)
    client = StubJSONClient(["not json", '{"ok": true}'], tracer)

    parsed = client.chat_json(
        prompt_name="rollout_reflection",
        messages=[{"role": "user", "content": "return json"}],
        max_tokens=768,
    )

    assert parsed == {"ok": True}
    assert [call["max_tokens"] for call in client.chat_calls] == [768, 768]


class FakeUsage:
    def model_dump(self):
        return {"total_tokens": 3}


class FakeCompletionsEndpoint:
    def __init__(self, text: str):
        self.text = text
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(text=self.text)],
            usage=FakeUsage(),
        )


class FakeChatEndpoint:
    def __init__(self, text: str):
        self.text = text
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))],
            usage=FakeUsage(),
        )


class StubCompleteClient(LLMClient):
    def __init__(self, config: LLMConfig, tracer: TraceWriter | None = None):
        self.config = config
        self.tracer = tracer
        self.completions_endpoint = FakeCompletionsEndpoint("raw completion")
        self.chat_endpoint = FakeChatEndpoint("raw chat completion")
        self._client = SimpleNamespace(
            completions=self.completions_endpoint,
            chat=SimpleNamespace(completions=self.chat_endpoint),
        )


def test_complete_uses_completions_endpoint_and_traces(tmp_path):
    tracer = TraceWriter(run_dir=tmp_path / "run", enabled=True)
    client = StubCompleteClient(
        LLMConfig(
            model="demo-model",
            turn_transport="completion",
            turn_temperature=0.4,
            turn_max_tokens=24,
        ),
        tracer,
    )

    text = client.complete(
        prompt_name="imagine_p2_reply",
        prompt="P2: ",
        state_id="state",
        stop=["\nP1:"],
    )

    assert text == "raw completion"
    assert client.completions_endpoint.kwargs["model"] == "demo-model"
    assert client.completions_endpoint.kwargs["prompt"] == "P2: "
    assert client.completions_endpoint.kwargs["temperature"] == 0.4
    assert client.completions_endpoint.kwargs["max_tokens"] == 24
    assert client.completions_endpoint.kwargs["stop"] == ["\nP1:"]
    events = [json.loads(line) for line in tracer.calls_path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["prompt_name"] == "imagine_p2_reply"
    assert events[-1]["raw_response"] == "raw completion"


def test_complete_can_use_chat_transport():
    client = StubCompleteClient(LLMConfig(model="demo-model", turn_transport="chat"))

    text = client.complete(prompt_name="realize_p1_move", prompt="P1: ", max_tokens=12)

    assert text == "raw chat completion"
    assert client.chat_endpoint.kwargs["messages"] == [{"role": "user", "content": "P1: "}]
    assert client.chat_endpoint.kwargs["max_tokens"] == 12
