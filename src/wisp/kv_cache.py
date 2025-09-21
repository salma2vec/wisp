from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch

@dataclass
class KVPage:
    key: torch.Tensor
    value: torch.Tensor
    used: int = 0

class PagedKVCache:
    def __init__(self, num_layers: int, num_heads: int, head_dim: int, page_size: int, device: str):
        self.num_layers = num_layers
        self.page_size = page_size
        self.device = device
        self.pages: Dict[Tuple[int,int], KVPage] = {}
        self.num_heads = num_heads
        self.head_dim = head_dim

    def new_sequence(self, seq_id: int):
        for l in range(self.num_layers):
            k = torch.empty(self.page_size, self.num_heads, self.head_dim, device=self.device)
            v = torch.empty_like(k)
            self.pages[(seq_id, l)] = KVPage(k, v, 0)

    def append(self, seq_id: int, layer: int, k_new: torch.Tensor, v_new: torch.Tensor):
        page = self.pages[(seq_id, layer)]
        n = k_new.shape[0]
        assert page.used + n <= self.page_size, "out of page capacity"
        page.key[page.used:page.used+n].copy_(k_new)
        page.value[page.used:page.used+n].copy_(v_new)
        page.used += n

    def view(self, seq_id: int, layer: int):
        page = self.pages[(seq_id, layer)]
        return page.key[:page.used], page.value[:page.used]

    def reset(self, seq_id: int):
        for l in range(self.num_layers):
            self.pages.pop((seq_id, l), None)
