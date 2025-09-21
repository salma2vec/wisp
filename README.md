# Wisp — a smol‑vLLM you can hack on

**Wisp** is a tiny LLM inference engine with just‑enough features to learn, extend, and ship:

- Batched prefill/ decode loop
- Paged KV cache (teaching‑grade)
- Eager FIFO scheduler
- Streaming‑ready FastAPI server
- HF model loader + optional `torch.compile`
- Top‑k/ top‑p/ temperature/ repetition penalty

## Quickstart
```bash
pip install -e .
python examples/serve.py  # starts FastAPI on :8000
curl -X POST localhost:8000/generate -H 'Content-Type: application/json'   -d '{"prompts":["hello"],"params":{"max_tokens":16}}'
```

## Design
*Small but honest.* The core loop is in `engine.py` so you can reason about latency, batching, and cache.

## Roadmap
- [ ] True paged KV across sequences
- [ ] Continuous batching & scheduling policies
- [ ] Tensor parallel (single host)

## License
MIT
