from __future__ import annotations
from typing import Tuple
import torch
from transformers import AutoModelForCausalLM

def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def maybe_compile(model: torch.nn.Module, enabled: bool):
    if enabled and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[arg-type]
    return model

def load_model(path: str, dtype: str = "auto", device: str = "auto") -> Tuple[torch.nn.Module, str]:
    dev = resolve_device(device)
    torch_dtype = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype)
    return model.to(dev), dev
