from __future__ import annotations
from collections import deque

class Request:
    def __init__(self, req_id: int, input_ids, params):
        self.id = req_id
        self.input_ids = input_ids
        self.params = params
        self.generated = []
        self.done = False

class EagerScheduler:
    def __init__(self):
        self.q = deque()
        self.next_id = 0
    def add(self, input_ids, params) -> int:
        rid = self.next_id; self.next_id += 1
        self.q.append(Request(rid, input_ids, params))
        return rid
    def pop_active(self, max_batch: int):
        batch = []
        while self.q and len(batch) < max_batch:
            r = self.q.popleft()
            if not r.done:
                batch.append(r)
        return batch
    def requeue(self, reqs):
        for r in reqs:
            if not r.done:
                self.q.append(r)
