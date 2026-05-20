from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from llm_mcts.web_launcher import WebLaunchConfig, WebLauncher


class FakeProcess:
    def __init__(self, returncode=None):
        self.signals = []
        self.killed = False
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def send_signal(self, signal):
        self.signals.append(signal)

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


def config(tmp_path: Path, **updates) -> WebLaunchConfig:
    game = tmp_path / "game.yaml"
    game.write_text("metadata: {name: fake}\n", encoding="utf-8")
    base = {
        "game": game,
        "model": "demo-model",
        "preset": None,
        "base_url": "http://127.0.0.1:8000/v1",
        "mock_llm": False,
        "llm_port": 8000,
        "api_port": 8787,
        "web_port": 5173,
        "max_simulations": 8,
        "no_vllm": False,
        "open_browser": False,
    }
    base.update(updates)
    return WebLaunchConfig(**base)


def test_ensure_vllm_reuses_ready_server(tmp_path, monkeypatch):
    launcher = WebLauncher(config(tmp_path))
    monkeypatch.setattr(launcher, "_is_llm_ready", lambda base_url: True)

    launcher._ensure_vllm("http://127.0.0.1:8000/v1")

    assert launcher.processes == []


def test_ensure_vllm_spawns_when_not_ready(tmp_path, monkeypatch):
    launcher = WebLauncher(config(tmp_path))
    readiness = iter([False, True])
    fake = FakeProcess()
    captured = {}
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/vllm")
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: captured.setdefault("args", args)
        and captured.setdefault("kwargs", kwargs)
        and fake,
    )
    monkeypatch.setattr(launcher, "_is_llm_ready", lambda base_url: next(readiness))

    launcher._ensure_vllm("http://127.0.0.1:8000/v1")

    assert launcher.processes[0].name == "vllm"
    assert captured["args"][0][:4] == ["/usr/bin/vllm", "serve", "demo-model", "--host"]


def test_ensure_vllm_requires_executable(tmp_path, monkeypatch):
    launcher = WebLauncher(config(tmp_path))
    monkeypatch.setattr(launcher, "_is_llm_ready", lambda base_url: False)
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="vllm executable not found"):
        launcher._ensure_vllm("http://127.0.0.1:8000/v1")


def test_wait_for_llm_fails_fast_when_vllm_exits(tmp_path, monkeypatch):
    launcher = WebLauncher(config(tmp_path))
    monkeypatch.setattr(launcher, "_is_llm_ready", lambda base_url: False)

    with pytest.raises(RuntimeError, match="vLLM exited"):
        launcher._wait_for_llm("http://127.0.0.1:8000/v1", process=FakeProcess(returncode=1))


def test_2080ti_qwen_preset_sets_model_args_and_cuda_device(tmp_path, monkeypatch):
    launcher = WebLauncher(config(tmp_path, model=None, preset="2080ti-qwen-awq"))
    readiness = iter([False, True])
    fake = FakeProcess()
    captured = {}
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/vllm")
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: captured.setdefault("args", args)
        and captured.setdefault("kwargs", kwargs)
        and fake,
    )
    monkeypatch.setattr(launcher, "_is_llm_ready", lambda base_url: next(readiness))

    launcher._ensure_vllm("http://127.0.0.1:8000/v1")

    command = captured["args"][0]
    assert command[:3] == ["/usr/bin/vllm", "serve", "Qwen/Qwen2.5-7B-Instruct-AWQ"]
    assert "--quantization" in command
    assert "awq" in command
    assert "--max-model-len" in command
    assert "8192" in command
    assert captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert captured["kwargs"]["env"]["HF_HUB_ENABLE_HF_TRANSFER"] == "1"


def test_2080ti_qwen_smoke_preset_sets_small_model(tmp_path, monkeypatch):
    launcher = WebLauncher(config(tmp_path, model=None, preset="2080ti-qwen-smoke"))
    readiness = iter([False, True])
    fake = FakeProcess()
    captured = {}
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/vllm")
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: captured.setdefault("args", args)
        and captured.setdefault("kwargs", kwargs)
        and fake,
    )
    monkeypatch.setattr(launcher, "_is_llm_ready", lambda base_url: next(readiness))

    launcher._ensure_vllm("http://127.0.0.1:8000/v1")

    command = captured["args"][0]
    assert command[:3] == ["/usr/bin/vllm", "serve", "Qwen/Qwen2.5-0.5B-Instruct"]
    assert "--quantization" not in command
    assert captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "1"
