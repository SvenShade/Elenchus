from __future__ import annotations

import os
from pathlib import Path

from llm_mcts.web_app import WebRuntimeConfig, create_app


def _runtime_from_env() -> WebRuntimeConfig:
    return WebRuntimeConfig(
        game_path=Path(os.environ["LLM_MCTS_WEB_GAME"]),
        base_url=os.environ.get("LLM_MCTS_BASE_URL"),
        model=os.environ.get("LLM_MCTS_MODEL"),
        mock_llm=os.environ.get("LLM_MCTS_MOCK_LLM", "0") == "1",
        max_simulations=int(os.environ.get("LLM_MCTS_MAX_SIMULATIONS", "128")),
        web_origin=os.environ.get("LLM_MCTS_WEB_ORIGIN"),
    )


app = create_app(_runtime_from_env())
