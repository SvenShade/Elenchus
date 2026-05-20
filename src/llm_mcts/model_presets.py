from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelPreset:
    name: str
    model: str
    description: str
    cuda_visible_devices: str | None = None
    vllm_args: list[str] = field(default_factory=list)


MODEL_PRESETS: dict[str, ModelPreset] = {
    "2080ti-qwen-smoke": ModelPreset(
        name="2080ti-qwen-smoke",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        description="Very small Qwen2.5 instruct model for fast end-to-end local smoke tests.",
        cuda_visible_devices="1",
        vllm_args=[
            "--dtype",
            "half",
            "--max-model-len",
            "4096",
            "--gpu-memory-utilization",
            "0.70",
        ],
    ),
    "2080ti-qwen-awq": ModelPreset(
        name="2080ti-qwen-awq",
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        description="Single RTX 2080 Ti friendly Qwen2.5 7B AWQ setup with extra runtime headroom.",
        cuda_visible_devices="1",
        vllm_args=[
            "--dtype",
            "half",
            "--quantization",
            "awq",
            "--max-model-len",
            "8192",
            "--gpu-memory-utilization",
            "0.82",
            "--enforce-eager",
        ],
    ),
    "2080ti-qwen-awq-tp2": ModelPreset(
        name="2080ti-qwen-awq-tp2",
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        description="Two RTX 2080 Ti Qwen2.5 7B AWQ setup using tensor parallelism across NVLink.",
        cuda_visible_devices="0,1",
        vllm_args=[
            "--tensor-parallel-size",
            "2",
            "--dtype",
            "half",
            "--quantization",
            "awq",
            "--max-model-len",
            "8192",
            "--gpu-memory-utilization",
            "0.78",
            "--enforce-eager",
        ],
    ),
    "2080ti-qwen-vl3b-text": ModelPreset(
        name="2080ti-qwen-vl3b-text",
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        description=(
            "Cached Qwen2.5-VL 3B setup served as text-only with multimodal "
            "slots disabled; useful when the 7B AWQ weights are not cached yet."
        ),
        cuda_visible_devices="1",
        vllm_args=[
            "--dtype",
            "half",
            "--max-model-len",
            "8192",
            "--gpu-memory-utilization",
            "0.88",
            "--limit-mm-per-prompt",
            '{"image":0,"video":0}',
        ],
    ),
}


def get_model_preset(name: str | None) -> ModelPreset | None:
    if name is None:
        return None
    return MODEL_PRESETS.get(name)


def preset_names() -> str:
    return ", ".join(sorted(MODEL_PRESETS))
