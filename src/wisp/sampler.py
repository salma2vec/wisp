from __future__ import annotations
import torch

def apply_repetition_penalty(logits, generated_ids, penalty: float):
    if penalty == 1.0 or generated_ids.numel() == 0:
        return logits
    logits[generated_ids] /= penalty
    return logits

def sample_logits(logits: torch.Tensor, temperature: float, top_p: float, top_k: int):
    logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = -float('inf')
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        mask = cum > top_p
        if mask.any():
            first = torch.argmax(mask.int()).item()
            sorted_logits[first+1:] = -float('inf')
        logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
