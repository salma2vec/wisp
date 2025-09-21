from wisp import Engine, EngineConfig, SamplingParams

def test_basic():
    e = Engine(EngineConfig(model_name_or_path="sshleifer/tiny-gpt2"))
    out = e.generate(["hi"], SamplingParams(max_tokens=8))
    assert isinstance(out[0], str)
