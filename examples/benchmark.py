from time import time
from wisp import Engine, EngineConfig, SamplingParams

engine = Engine(EngineConfig(model_name_or_path="gpt2", compile=False))

prompts = ["Hello world"] * 8
start = time()
outs = engine.generate(prompts, SamplingParams(max_tokens=64))
print("throughput toks/s ~", (8*64)/(time()-start))
