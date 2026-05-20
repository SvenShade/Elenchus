from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from rich.console import Console

from llm_mcts.model_presets import get_model_preset, preset_names


@dataclass
class WebLaunchConfig:
    game: Path
    model: str | None
    preset: str | None
    base_url: str | None
    mock_llm: bool
    llm_port: int
    api_port: int
    web_port: int
    max_simulations: int
    no_vllm: bool
    open_browser: bool
    cuda_visible_devices: str | None = None
    vllm_dtype: str | None = None
    vllm_quantization: str | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    tensor_parallel_size: int | None = None
    llm_timeout_s: float = 600.0
    vllm_extra_args: list[str] = field(default_factory=list)


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen


@dataclass
class WebLauncher:
    config: WebLaunchConfig
    console: Console = field(default_factory=Console)
    processes: list[ManagedProcess] = field(default_factory=list)

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def run(self) -> None:
        base_url = self.config.base_url or f"http://127.0.0.1:{self.config.llm_port}/v1"
        self._validate_preset()
        try:
            if not self.config.mock_llm and not self.config.no_vllm:
                self._ensure_vllm(base_url)
            elif self.config.no_vllm and not self.config.mock_llm:
                self.console.print(f"[yellow]Skipping vLLM launch; API will use {base_url}[/yellow]")

            self._ensure_frontend_dependencies()
            api_process = self._start_api(base_url)
            self._wait_http(f"http://127.0.0.1:{self.config.api_port}/api/health")
            web_process = self._start_frontend()
            url = f"http://127.0.0.1:{self.config.web_port}"
            self._wait_http(url)
            self.console.print(f"[bold green]llm-mcts web is running[/bold green]: {url}")
            if self.config.open_browser:
                webbrowser.open(url)
            self._wait_forever([api_process, web_process])
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Stopping llm-mcts web...[/yellow]")
        finally:
            self.stop_all()

    def _ensure_vllm(self, base_url: str) -> None:
        if self._is_llm_ready(base_url):
            self.console.print(f"[green]Reusing local LLM server at {base_url}[/green]")
            return
        model = self._resolved_model()
        if not model:
            raise RuntimeError("--model is required when launching vLLM")
        vllm = shutil.which("vllm")
        if vllm is None:
            raise RuntimeError(
                "vllm executable not found. Install with "
                "`uv sync --extra web --extra local-llm`, pass --mock-llm, or use --no-vllm."
            )
        env = os.environ.copy()
        cuda_visible_devices = self._resolved_cuda_visible_devices()
        if cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            self.console.print(f"[cyan]Using CUDA_VISIBLE_DEVICES={cuda_visible_devices}[/cyan]")
        env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        process = subprocess.Popen(
            self._vllm_command(vllm, model),
            env=env,
            cwd=self.repo_root,
        )
        self.processes.append(ManagedProcess("vllm", process))
        self.console.print(f"[cyan]Started vLLM on {base_url}; waiting for readiness...[/cyan]")
        self._wait_for_llm(
            base_url,
            timeout_s=self.config.llm_timeout_s,
            process=process,
        )

    def _start_api(self, base_url: str) -> ManagedProcess:
        env = os.environ.copy()
        env.update(
            {
                "LLM_MCTS_WEB_GAME": str(self.config.game.resolve()),
                "LLM_MCTS_BASE_URL": base_url,
                "LLM_MCTS_MAX_SIMULATIONS": str(self.config.max_simulations),
                "LLM_MCTS_MOCK_LLM": "1" if self.config.mock_llm else "0",
                "LLM_MCTS_WEB_ORIGIN": f"http://127.0.0.1:{self.config.web_port}",
            }
        )
        model = self._resolved_model()
        if model:
            env["LLM_MCTS_MODEL"] = model
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "llm_mcts.web_server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.config.api_port),
            ],
            env=env,
            cwd=self.repo_root,
        )
        managed = ManagedProcess("api", process)
        self.processes.append(managed)
        return managed

    def _start_frontend(self) -> ManagedProcess:
        frontend = self.repo_root / "frontend"
        env = os.environ.copy()
        env["VITE_API_BASE"] = f"http://127.0.0.1:{self.config.api_port}"
        process = subprocess.Popen(
            [
                "npm",
                "--prefix",
                str(frontend),
                "run",
                "dev",
                "--",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.config.web_port),
            ],
            env=env,
            cwd=self.repo_root,
        )
        managed = ManagedProcess("frontend", process)
        self.processes.append(managed)
        return managed

    def _ensure_frontend_dependencies(self) -> None:
        frontend = self.repo_root / "frontend"
        if not (frontend / "node_modules").exists():
            self.console.print("[cyan]Installing frontend dependencies...[/cyan]")
            subprocess.run(
                ["npm", "--prefix", str(frontend), "install"],
                cwd=self.repo_root,
                check=True,
            )

    def _wait_forever(self, required: list[ManagedProcess]) -> None:
        while True:
            for managed in required:
                code = managed.process.poll()
                if code is not None:
                    raise RuntimeError(f"{managed.name} exited with status {code}")
            time.sleep(0.5)

    def _wait_for_llm(
        self,
        base_url: str,
        timeout_s: float = 600.0,
        process: subprocess.Popen | None = None,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._is_llm_ready(base_url):
                self.console.print(f"[green]Local LLM server is ready at {base_url}[/green]")
                return
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"vLLM exited before becoming ready with status {process.returncode}")
            time.sleep(2.0)
        raise TimeoutError(f"Timed out waiting for local LLM server at {base_url}")

    def _wait_http(self, url: str, timeout_s: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                response = httpx.get(url, timeout=1.0)
                if response.status_code < 500:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        raise TimeoutError(f"Timed out waiting for {url}")

    def _is_llm_ready(self, base_url: str) -> bool:
        try:
            response = httpx.get(f"{base_url.rstrip('/')}/models", timeout=1.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def _validate_preset(self) -> None:
        if self.config.preset and get_model_preset(self.config.preset) is None:
            raise RuntimeError(
                f"Unknown model preset '{self.config.preset}'. Available presets: {preset_names()}"
            )

    def _resolved_model(self) -> str | None:
        preset = get_model_preset(self.config.preset)
        return self.config.model or (preset.model if preset else None)

    def _resolved_cuda_visible_devices(self) -> str | None:
        preset = get_model_preset(self.config.preset)
        return self.config.cuda_visible_devices or (
            preset.cuda_visible_devices if preset else None
        )

    def _vllm_command(self, vllm: str, model: str) -> list[str]:
        command = [
            vllm,
            "serve",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.config.llm_port),
        ]
        preset = get_model_preset(self.config.preset)
        if preset:
            command.extend(preset.vllm_args)
        if self.config.vllm_dtype:
            command.extend(["--dtype", self.config.vllm_dtype])
        if self.config.vllm_quantization:
            command.extend(["--quantization", self.config.vllm_quantization])
        if self.config.max_model_len is not None:
            command.extend(["--max-model-len", str(self.config.max_model_len)])
        if self.config.gpu_memory_utilization is not None:
            command.extend(
                ["--gpu-memory-utilization", str(self.config.gpu_memory_utilization)]
            )
        if self.config.tensor_parallel_size is not None:
            command.extend(["--tensor-parallel-size", str(self.config.tensor_parallel_size)])
        command.extend(self.config.vllm_extra_args)
        return command

    def stop_all(self) -> None:
        for managed in reversed(self.processes):
            if managed.process.poll() is None:
                self.console.print(f"[dim]Stopping {managed.name}[/dim]")
                managed.process.send_signal(signal.SIGTERM)
        for managed in reversed(self.processes):
            try:
                managed.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                managed.process.kill()
