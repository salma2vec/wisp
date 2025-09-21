from __future__ import annotations
from typing import List
import torch
from .config import EngineConfig, SamplingParams
from .tokenizer import Tokenizer
from .model_loader import load_model, maybe_compile
from .kv_cache import PagedKVCache
from .sampler import sample_logits, apply_repetition_penalty
from .scheduler import EagerScheduler

class Engine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.model, self.device = load_model(cfg.model_name_or_path, cfg.dtype, cfg.device)
        self.model = maybe_compile(self.model, cfg.compile)
        self.model.eval()
        self.tok = Tokenizer(cfg.model_name_or_path)
        n_layer = getattr(self.model.config, 'num_hidden_layers', 24)
        n_head = getattr(self.model.config, 'num_attention_heads', 16)
        head_dim = getattr(self.model.config, 'hidden_size', 4096) // n_head
        self.kv = PagedKVCache(n_layer, n_head, head_dim, cfg.page_size, self.device)
        self.sched = EagerScheduler()

    @torch.no_grad()
    def generate(self, prompts: List[str], params: SamplingParams):
        ids = [self.tok.encode(p).input_ids.to(self.device) for p in prompts]
        req_ids = [self.sched.add(x, params) for x in ids]
        results = {rid: "" for rid in req_ids}

        active = self.sched.pop_active(max_batch=8)
        seq_ids = list(range(len(active)))
        for sid in seq_ids:
            self.kv.new_sequence(sid)

        while active:
            max_len = max(r.input_ids.shape[-1] for r in active)
            padded = []
            attn_mask = []
            for r in active:
                pad = max_len - r.input_ids.shape[-1]
                if pad:
                    pad_ids = torch.full((1, pad), self.tok.eos_token_id, device=self.device)
                    x = torch.cat([r.input_ids, pad_ids], dim=-1)
                else:
                    x = r.input_ids
                padded.append(x)
                attn_mask.append(torch.ones_like(x))
            batch = torch.cat(padded, dim=0)

            out = self.model(input_ids=batch, attention_mask=torch.cat(attn_mask, dim=0))
            logits = out.logits[:, -1, :]

            for i, r in enumerate(active):
                last = logits[i]
                if r.generated:
                    last = apply_repetition_penalty(last, torch.tensor(r.generated, device=last.device), r.params.repetition_penalty)
                token = sample_logits(last, r.params.temperature, r.params.top_p, r.params.top_k)
                r.generated.append(token)
                r.input_ids = torch.cat([r.input_ids, torch.tensor([[token]], device=self.device)], dim=-1)
                if token == self.tok.eos_token_id or len(r.generated) >= r.params.max_tokens:
                    r.done = True

            for r in active:
                ridx = req_ids[active.index(r)]
                results[ridx] = self.tok.decode(r.input_ids[:, -len(r.generated):])[0]

            self.sched.requeue(active)
            active = self.sched.pop_active(max_batch=8)

        return [results[rid] for rid in req_ids]
