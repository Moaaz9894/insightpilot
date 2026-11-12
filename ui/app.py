"""Chat UI (Chainlit).

Run:  chainlit run ui/app.py -w
Env:  API_URL (default http://localhost:8000), API_KEY (optional)
"""
import asyncio
import json
import os

import chainlit as cl
import requests

API = os.getenv("API_URL", "http://localhost:8000")
HEADERS = {"x-api-key": os.environ["API_KEY"]} if os.getenv("API_KEY") else {}


def call_api(question: str, history: list[dict]) -> dict:
    """Blocking call to the existing /chat endpoint (run in a worker thread)."""
    r = requests.post(
        f"{API}/chat",
        json={"question": question, "history": history},
        headers=HEADERS,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def to_text(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(value)


@cl.set_starters
async def starters():
    return [
        cl.Starter(label="Forecast", message="Forecast BEVERAGES in Guayas for 14 days"),
        cl.Starter(label="Sales", message="What were the top 5 product families by sales last month?"),
        cl.Starter(label="Policies", message="What is the company's return policy?"),
    ]


@cl.on_chat_start
async def on_start():
    cl.user_session.set("history", [])


@cl.on_message
async def on_message(message: cl.Message):
    history: list[dict] = cl.user_session.get("history")

    # 1) call the API without blocking the UI
    try:
        async with cl.Step(name="InsightPilot agent", type="run") as run:
            data = await cl.make_async(call_api)(message.content, history[-6:])
            run.output = f"mode: {data.get('mode')} · {data.get('latency_ms')} ms"
    except Exception as e:
        await cl.Message(content=f"⚠️ API error: `{e}`").send()
        return

    # 2) show every tool call as its own collapsible step
    for tc in data.get("tool_calls") or []:
        if not isinstance(tc, dict):
            tc = {"result": tc}
        name = tc.get("name") or tc.get("tool") or "tool"
        args = tc.get("args") or tc.get("arguments") or tc.get("input")
        result = tc.get("result") or tc.get("output")
        async with cl.Step(name=str(name), type="tool", language="json") as step:
            if args is not None:
                step.input = to_text(args)
            step.output = to_text(result if result is not None else tc)

    # 3) type the answer out progressively
    answer = data.get("answer", "")
    out = cl.Message(content="")
    for i in range(0, len(answer), 6):
        await out.stream_token(answer[i : i + 6])
        await asyncio.sleep(0.008)
    await out.send()

    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": answer})
    cl.user_session.set("history", history)