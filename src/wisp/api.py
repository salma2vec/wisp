from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from .config import EngineConfig, SamplingParams
from .engine import Engine

app = FastAPI(title="Wisp")
engine = None

class GenReq(BaseModel):
    prompts: list[str]
    params: SamplingParams | None = None

@app.on_event("startup")
def startup():
    global engine
    engine = Engine(EngineConfig(model_name_or_path="gpt2"))

@app.post("/generate")
def generate(req: GenReq):
    params = req.params or SamplingParams()
    return {"outputs": engine.generate(req.prompts, params)}
