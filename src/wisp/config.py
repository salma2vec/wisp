from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0

@dataclass
class EngineConfig:
    model_name_or_path: str = "gpt2"
    dtype: str = "auto"  # "auto", "float16", "bfloat16"
    device: str = "auto"  # "cuda" if available else "cpu"
    compile: bool = False
    max_batch_tokens: int = 8192
    page_size: int = 2048  # tokens per KV page
