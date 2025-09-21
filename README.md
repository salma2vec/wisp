# Wisp 

A tiny LLM inference engine with just‑enough features to learn, extend, and ship:

- Batched prefill/ decode loop
- Paged KV cache (teaching‑grade)
- Eager FIFO scheduler
- Streaming‑ready FastAPI server
- HF model loader + optional `torch.compile`
- Top‑k/ top‑p/ temperature/ repetition penalty

## Design
The core loop is in `engine.py` so you can reason about latency, batching, and cache.

## Roadmap
- [ ] True paged KV across sequences
- [ ] Continuous batching & scheduling policies
- [ ] Tensor parallel (single host)

## License
MIT
