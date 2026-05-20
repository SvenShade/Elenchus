from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_mcts.state import TranscriptTurn


class MetadataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    max_real_turns: int = Field(default=1, ge=1)


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "local"
    model: str = "local-model"
    temperature: float = Field(default=0.2, ge=0.0)
    max_tokens: int = Field(default=512, ge=1)
    timeout: float = Field(default=60.0, gt=0.0)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    turn_transport: Literal["completion", "chat"] = "completion"
    turn_max_tokens: int = Field(default=96, ge=1)
    turn_temperature: float | None = Field(default=None, ge=0.0)


class PlayerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    goals: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    hierarchy: str | None = None
    private_context: dict[str, Any] | list[Any] | str = Field(default_factory=dict)
    communication_style: str = ""
    beliefs_about_teammate: dict[str, Any] | list[Any] | str = Field(default_factory=dict)


class ActionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    allowed_players: list[Literal["P1", "P2"]] = Field(default_factory=lambda: ["P1"])
    realization_guidance: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_prior: str
    realize_p1_move: str
    imagine_p2_reply: str
    judge_rollout_state: str
    actual_p2_reply: str
    state_analysis: str | None = None
    candidate_actions: str | None = None
    exploration_summary: str | None = None
    rollout_reflection: str | None = None
    finalize_p1_move: str | None = None


class ProgressiveWideningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    k: float = Field(default=3.0, gt=0.0)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    initial_children: int = Field(default=4, ge=1)
    expand_batch: int = Field(default=1, ge=1)
    max_candidates: int = Field(default=40, ge=1)
    candidate_refresh_visits: list[int] = Field(default_factory=lambda: [0, 5, 20, 60])


class MCTSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    simulations: int = Field(default=16, ge=1)
    max_rollout_depth: int = Field(default=2, ge=1)
    c_puct: float = Field(default=1.5, ge=0.0)
    root_action_selection: Literal["visits", "value"] = "visits"
    random_seed: int = 0
    trace: bool = True
    progressive_widening: ProgressiveWideningConfig = Field(default_factory=ProgressiveWideningConfig)


class InitialStateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    transcript: list[TranscriptTurn] = Field(default_factory=list)


class GameConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: MetadataConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)
    players: dict[Literal["P1", "P2"], PlayerConfig]
    actions: list[ActionConfig]
    prompts: PromptConfig
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    initial_state: InitialStateConfig

    @classmethod
    def load(cls, path: str | Path) -> "GameConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls.model_validate(data)

    @model_validator(mode="after")
    def validate_game(self) -> "GameConfig":
        if "P1" not in self.players or "P2" not in self.players:
            raise ValueError("players must define both P1 and P2")
        if not self.actions:
            raise ValueError("at least one action is required")
        action_ids = [action.id for action in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("action ids must be unique")
        p1_actions = [action for action in self.actions if "P1" in action.allowed_players]
        if not p1_actions:
            raise ValueError("at least one action must be legal for P1")
        return self
