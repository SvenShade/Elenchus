from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, run_dir: str | Path | None = None, enabled: bool = True):
        self.enabled = enabled
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(run_dir) if run_dir else Path("runs") / timestamp
        self.calls_path = self.run_dir / "calls.jsonl"
        self.tree_path = self.run_dir / "tree.json"
        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def record_call(
        self,
        *,
        prompt_name: str,
        messages: list[dict[str, str]],
        raw_response: str | None,
        parsed: Any = None,
        error: str | None = None,
        latency_s: float | None = None,
        state_id: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        event = {
            "prompt_name": prompt_name,
            "state_id": state_id,
            "latency_s": latency_s,
            "messages": messages,
            "raw_response": raw_response,
            "parsed": parsed,
            "error": error,
            "usage": usage,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with self.calls_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True, default=str) + "\n")

    def save_tree(self, tree: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self.tree_path.open("w", encoding="utf-8") as handle:
            json.dump(tree, handle, indent=2, ensure_ascii=True, default=str)
