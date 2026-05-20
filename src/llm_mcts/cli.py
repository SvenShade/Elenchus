from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_mcts.config import GameConfig, LLMConfig
from llm_mcts.env import TwoPlayerConversationEnv
from llm_mcts.llm import LLMClient
from llm_mcts.mcts import MCTSPlanner, PlanResult
from llm_mcts.mock import DeterministicMockLLM
from llm_mcts.trace import TraceWriter

app = typer.Typer(no_args_is_help=True)
console = Console()


class P2Mode(str, Enum):
    llm = "llm"
    manual = "manual"


@app.command()
def validate(
    game: Annotated[Path, typer.Option("--game", exists=True, dir_okay=False)],
) -> None:
    """Validate a game YAML file."""
    config = GameConfig.load(game)
    console.print(
        Panel.fit(
            f"[bold green]Valid game[/bold green]\n{config.metadata.name}\n"
            f"{len(config.actions)} actions, {config.mcts.simulations} simulations",
            title="llm-mcts",
        )
    )


@app.command()
def plan(
    game: Annotated[Path, typer.Option("--game", exists=True, dir_okay=False)],
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    mock_llm: Annotated[
        bool,
        typer.Option("--mock-llm", help="Use a deterministic fake LLM instead of HTTP."),
    ] = False,
    run_dir: Annotated[Path | None, typer.Option("--run-dir")] = None,
) -> None:
    """Plan one P1 move from the initial state."""
    config, tracer, env = _build_runtime(game, base_url, model, mock_llm, run_dir)
    planner = MCTSPlanner(env, tracer=tracer)
    result = planner.plan(env.initial_state())
    _print_plan_result(result)


@app.command()
def play(
    game: Annotated[Path, typer.Option("--game", exists=True, dir_okay=False)],
    p2_mode: Annotated[P2Mode, typer.Option("--p2-mode")] = P2Mode.llm,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    mock_llm: Annotated[
        bool,
        typer.Option("--mock-llm", help="Use a deterministic fake LLM instead of HTTP."),
    ] = False,
    run_dir: Annotated[Path | None, typer.Option("--run-dir")] = None,
) -> None:
    """Run a short P1/P2 conversation."""
    config, tracer, env = _build_runtime(game, base_url, model, mock_llm, run_dir)
    planner = MCTSPlanner(env, tracer=tracer)
    state = env.initial_state()

    for turn in range(config.metadata.max_real_turns):
        console.rule(f"Turn {turn + 1}")
        result = planner.plan(state)
        env.remember_rollout_reflection(result.rollout_reflection)
        _print_plan_result(result)
        state = state.append("P1", result.p1_utterance, action_id=result.chosen_action_id)
        env.remember_cognitive_state(state)
        console.print(f"[bold]P1[/bold]: {result.p1_utterance}")

        if p2_mode == P2Mode.manual:
            response = typer.prompt("P2")
            state = env.append_manual_p2_response(state, response)
        else:
            response = env.actual_p2_reply(state)
            state = state.append("P2", response)
            env.remember_cognitive_state(state)
        console.print(f"[bold]P2[/bold]: {response}")

    if tracer.enabled:
        console.print(f"[dim]Trace directory:[/dim] {tracer.run_dir}")


@app.command()
def web(
    game: Annotated[Path, typer.Option("--game", exists=True, dir_okay=False)],
    model: Annotated[str | None, typer.Option("--model")] = None,
    preset: Annotated[
        str | None,
        typer.Option(
            "--preset",
            help="Hardware/model preset. Currently: 2080ti-qwen-smoke, 2080ti-qwen-awq.",
        ),
    ] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    mock_llm: Annotated[
        bool,
        typer.Option("--mock-llm", help="Use a deterministic fake LLM instead of HTTP."),
    ] = False,
    llm_port: Annotated[int, typer.Option("--llm-port")] = 8000,
    api_port: Annotated[int, typer.Option("--api-port")] = 8787,
    web_port: Annotated[int, typer.Option("--web-port")] = 5173,
    max_simulations: Annotated[int, typer.Option("--max-simulations")] = 128,
    no_vllm: Annotated[
        bool,
        typer.Option("--no-vllm", help="Use the configured base URL without launching vLLM."),
    ] = False,
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
    cuda_visible_devices: Annotated[str | None, typer.Option("--cuda-visible-devices")] = None,
    vllm_dtype: Annotated[str | None, typer.Option("--vllm-dtype")] = None,
    vllm_quantization: Annotated[str | None, typer.Option("--vllm-quantization")] = None,
    max_model_len: Annotated[int | None, typer.Option("--max-model-len")] = None,
    gpu_memory_utilization: Annotated[
        float | None,
        typer.Option("--gpu-memory-utilization"),
    ] = None,
    tensor_parallel_size: Annotated[int | None, typer.Option("--tensor-parallel-size")] = None,
    llm_timeout_s: Annotated[float, typer.Option("--llm-timeout-s")] = 600.0,
    vllm_extra_args: Annotated[
        list[str] | None,
        typer.Option("--vllm-arg", help="Additional raw argument to append to vLLM."),
    ] = None,
) -> None:
    """Start the visual web UI and, by default, a local vLLM server."""
    from llm_mcts.web_launcher import WebLaunchConfig, WebLauncher

    WebLauncher(
        WebLaunchConfig(
            game=game,
            model=model,
            preset=preset,
            base_url=base_url,
            mock_llm=mock_llm,
            llm_port=llm_port,
            api_port=api_port,
            web_port=web_port,
            max_simulations=max_simulations,
            no_vllm=no_vllm,
            open_browser=open_browser,
            cuda_visible_devices=cuda_visible_devices,
            vllm_dtype=vllm_dtype,
            vllm_quantization=vllm_quantization,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            llm_timeout_s=llm_timeout_s,
            vllm_extra_args=vllm_extra_args or [],
        )
    ).run()


def _build_runtime(
    game: Path,
    base_url: str | None,
    model: str | None,
    mock_llm: bool,
    run_dir: Path | None,
) -> tuple[GameConfig, TraceWriter, TwoPlayerConversationEnv]:
    config = GameConfig.load(game)
    llm_updates = {}
    if base_url:
        llm_updates["base_url"] = base_url
    if model:
        llm_updates["model"] = model
    if llm_updates:
        config = config.model_copy(
            update={"llm": config.llm.model_copy(update=llm_updates)}
        )

    tracer = TraceWriter(run_dir=run_dir, enabled=config.mcts.trace)
    llm = DeterministicMockLLM(tracer=tracer) if mock_llm else LLMClient(config.llm, tracer=tracer)
    return config, tracer, TwoPlayerConversationEnv(config, llm)


def _print_plan_result(result: PlanResult) -> None:
    console.print(
        Panel(
            result.p1_utterance,
            title=f"Selected action: {result.chosen_action_id}",
            subtitle=f"estimated utility {result.estimated_utility:.3f}",
        )
    )
    table = Table(title="Root action stats")
    table.add_column("action")
    table.add_column("visits", justify="right")
    table.add_column("value", justify="right")
    table.add_column("prior", justify="right")
    for item in result.root_stats[:8]:
        table.add_row(
            str(item["action_id"]),
            str(item["visits"]),
            f"{float(item['value']):.3f}",
            f"{float(item['prior']):.3f}",
        )
    console.print(table)
    if result.trace_dir:
        console.print(f"[dim]Trace directory:[/dim] {result.trace_dir}")
