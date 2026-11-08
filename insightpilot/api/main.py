"""REST API. Run: uvicorn insightpilot.api.main:app --reload"""
from __future__ import annotations
import logging
import os
import time
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from ..agent import tools
from ..agent.agent import Agent
from ..bootstrap import ensure
from ..ml import predict

log = logging.getLogger("insightpilot")
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure()
    _state["agent"] = Agent()
    yield


app = FastAPI(title="InsightPilot", version="1.0.0", lifespan=lifespan)


def auth(x_api_key: str | None = Header(default=None)):
    """Optional API-key auth: enforced only when API_KEY is set (use Key Vault in Azure)."""
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(401, "invalid api key")


class ChatIn(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    history: list[dict] = []


@app.get("/health")
def health():
    return {"status": "ok", "mode": "azure-openai" if _state["agent"].online else "offline-demo",
            "model_val_wmape": predict.dimensions()["metrics"]["val_wmape"]}


@app.post("/chat", dependencies=[Depends(auth)])
def chat(body: ChatIn):
    try:
        out = _state["agent"].run(body.question, body.history)
    except Exception as e:
        log.exception("agent failure")
        raise HTTPException(500, f"agent error: {type(e).__name__}")
    log.info("chat mode=%s tools=%s latency_ms=%s", out["mode"],
             [c["tool"] for c in out["tool_calls"]], out["latency_ms"])
    return out


@app.get("/forecast", dependencies=[Depends(auth)])
def forecast(sku: str, region: str, horizon_days: int = 14, promo: bool = False):
    t0 = time.time()
    try:
        out = tools.get_forecast(sku, region, horizon_days, promo)
    except ValueError as e:
        raise HTTPException(422, str(e))
    out["latency_ms"] = int((time.time() - t0) * 1000)
    return out


@app.get("/policies/search", dependencies=[Depends(auth)])
def policies(q: str, k: int = 3):
    return tools.search_policies(q, k)
