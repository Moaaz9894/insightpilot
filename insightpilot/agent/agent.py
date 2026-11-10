"""Tool-calling agent.
 * Online mode: Azure OpenAI function calling loop (production path).
 * Offline mode: deterministic keyword router so the whole stack is demoable/testable with no keys."""
from __future__ import annotations
import json
import os
import re
import time
from ..config import azure_openai_configured
from ..data.generate import PRODUCTS
from ..ml import predict
from . import tools

SYSTEM_PROMPT = """You are InsightPilot, an assistant for sales, supply chain and revenue teams.
Rules:
- Facts about sales come ONLY from query_sales_sql. Never invent numbers.
- Forecasts come ONLY from get_forecast. Say they are model estimates.
- Company rules/limits come ONLY from search_policies; cite the policy name.
- If a tool errors, fix the input and retry once, otherwise say what failed.
- Answer concisely, with the key number first. Reply in the user's language."""

MAX_STEPS = 6


class Agent:
    def __init__(self):
        self.online = azure_openai_configured()
        if self.online:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"))
            self.model = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    # ------------------------------------------------------------------ public
    def run(self, question: str, history: list[dict] | None = None) -> dict:
        t0 = time.time()
        out = self._run_online(question, history or []) if self.online else self._run_offline(question)
        out["mode"] = "azure-openai" if self.online else "offline-demo"
        out["latency_ms"] = int((time.time() - t0) * 1000)
        return out

    # ------------------------------------------------------------------ online
    def _call_tool(self, name: str, args: dict) -> str:
        try:
            return json.dumps(tools.TOOL_FUNCS[name](**args), default=str)
        except Exception as e:  # return errors to the model so it can self-correct
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def _run_online(self, question: str, history: list[dict]) -> dict:
        messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + tools.get_schema()},
                    *history, {"role": "user", "content": question}]
        calls = []
        for _ in range(MAX_STEPS):
            # gpt-5-mini is a reasoning model: it rejects `temperature`, so none is passed.
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools.TOOL_SPECS)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return {"answer": msg.content, "tool_calls": calls}
            messages.append(msg.model_dump(exclude_none=True))
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = self._call_tool(tc.function.name, args)
                calls.append({"tool": tc.function.name, "args": args})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        return {"answer": "I could not finish within the step limit; please rephrase.", "tool_calls": calls}

    # ----------------------------------------------------------------- offline
    def _run_offline(self, q: str) -> dict:
        ql = q.lower()
        calls = []

        def call(name, **args):
            calls.append({"tool": name, "args": args})
            return tools.TOOL_FUNCS[name](**args)

        dims = predict.dimensions()
        # longest-first so 'GROCERY II' wins over 'GROCERY I'
        region = next((r for r in sorted(dims["regions"], key=len, reverse=True) if r.lower() in ql), None)
        sku = next((x for x in sorted(dims["skus"], key=len, reverse=True) if x.lower() in ql), None)
        m = re.search(r"sku-?(\d{3})", ql)
        if not sku and m and f"SKU-{m.group(1)}" in dims["skus"]:
            sku = f"SKU-{m.group(1)}"
        if not sku:  # synthetic demo data: match by product name
            sku = next((p[0] for p in PRODUCTS if p[0] in dims["skus"]
                        and (p[1].lower() in ql or p[1].split()[0].lower() in ql)), None)

        if re.search(r"forecast|predict|expect|توقع", ql) and sku and region:
            h = re.search(r"(\d+)\s*day", ql)
            r = call("get_forecast", sku=sku, region=region, horizon_days=min(int(h.group(1)), 28) if h else 14)
            return {"answer": f"Model forecast for {r['sku']} in {r['region']}: about {r['total_units']} units "
                              f"over the next {r['horizon_days']} days (validation WMAPE {r['model_val_wmape']:.1%}).",
                    "tool_calls": calls}
        if re.search(r"policy|rule|limit|allowed|discount|promo|return|credit|safety stock|expiry|price floor|سياسة", ql):
            r = call("search_policies", query=q)["results"]
            if r:
                top = r[0]
                return {"answer": f"{top['heading']}: {top['text']}", "tool_calls": calls, "sources": [x["id"] for x in r]}
        if re.search(r"top|best|highest|revenue|sales|units|إيراد|مبيعات", ql):
            by = "region" if "region" in ql else "category" if "categor" in ql else "product_name"
            r = call("query_sales_sql",
                     sql=f"SELECT {by}, SUM(units) AS units FROM gold_daily_sales "
                         f"GROUP BY 1 ORDER BY 2 DESC LIMIT 5")
            lines = ", ".join(f"{a}: {b:,.0f}" for a, b in r["rows"])
            return {"answer": f"Top {by} by units sold - {lines}", "tool_calls": calls}
        return {"answer": "Offline demo mode understands: forecasts (e.g. 'forecast <sku> in <region> for 14 days'), "
                          "policy questions, and sales rankings. Configure Azure OpenAI for free-form questions.",
                "tool_calls": calls}