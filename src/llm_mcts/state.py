from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict


class TranscriptTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker: str
    content: str
    action_id: str | None = None
    imagined: bool = False


@dataclass
class ConversationState:
    task: str
    transcript: list[TranscriptTurn] = field(default_factory=list)
    current_turn_index: int = 0
    terminal: bool = False

    def copy(self) -> "ConversationState":
        return ConversationState(
            task=self.task,
            transcript=[
                TranscriptTurn.model_validate(turn.model_dump()) for turn in self.transcript
            ],
            current_turn_index=self.current_turn_index,
            terminal=self.terminal,
        )

    def append(
        self,
        speaker: str,
        content: str,
        action_id: str | None = None,
        imagined: bool = False,
    ) -> "ConversationState":
        next_state = self.copy()
        next_state.transcript.append(
            TranscriptTurn(
                speaker=speaker,
                content=content,
                action_id=action_id,
                imagined=imagined,
            )
        )
        next_state.current_turn_index += 1
        return next_state

    def transcript_text(self, include_metadata: bool = True) -> str:
        if not self.transcript:
            return "(empty transcript)"
        lines = []
        for idx, turn in enumerate(self.transcript, start=1):
            prefix = f"{idx}. {turn.speaker}"
            if include_metadata and turn.action_id:
                prefix += f" [{turn.action_id}]"
            if include_metadata and turn.imagined:
                prefix += " (imagined)"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)

    def dialogue_text(self) -> str:
        if not self.transcript:
            return ""
        return "\n".join(f"{turn.speaker}: {turn.content}" for turn in self.transcript)

    def model_dump(self) -> dict[str, object]:
        return {
            "task": self.task,
            "transcript": [turn.model_dump() for turn in self.transcript],
            "current_turn_index": self.current_turn_index,
            "terminal": self.terminal,
        }

    def state_hash(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
