"""Offline evaluation. Run: python -m eval.run_eval   -> prints + writes eval/results.json
Measures: (1) retrieval quality, (2) forecast accuracy vs baseline, (3) SQL-guard coverage,
(4) agent tool-routing accuracy. With Azure OpenAI configured, extend `AGENT_CASES` and add an
LLM-as-judge for groundedness (see README 'Evaluation')."""
import json
from pathlib import Path
from insightpilot.bootstrap import ensure
from insightpilot.agent import tools
from insightpilot.agent.agent import Agent
from insightpilot.ml import predict
from insightpilot.rag.retriever import get_retriever

RETRIEVAL = [
    ("What is the maximum promotional discount?", "promotions_policy"),
    ("How long can a promotion run?", "promotions_policy"),
    ("Who approves a price change above 5 percent?", "pricing_policy"),
    ("What margin is required for the price floor?", "pricing_policy"),
    ("How many days of safety stock for fast moving items?", "supply_chain_policy"),
    ("What shelf life must products have when leaving the warehouse?", "supply_chain_policy"),
    ("How fast are urgent orders delivered?", "distribution_policy"),
    ("When is a route split?", "distribution_policy"),
    ("What are the standard payment terms?", "finance_returns_policy"),
    ("How soon must damaged goods be reported?", "finance_returns_policy"),
]
SQL_ATTACKS = ["DROP TABLE gold_daily_sales", "SELECT 1; SELECT 2", "SELECT * FROM read_csv('/etc/passwd')",
               "DELETE FROM gold_daily_sales", "SELECT 1 -- x", "COPY gold_daily_sales TO 'x.csv'"]
AGENT_CASES = [("forecast SKU-003 in South for 10 days", "get_forecast"),
               ("what is the credit limit policy", "search_policies"),
               ("top categories by revenue", "query_sales_sql")]


def main():
    ensure()
    r = get_retriever()
    hits1 = hits3 = 0
    for q, doc in RETRIEVAL:
        docs = [x["doc"] for x in r.search(q, 3)]
        hits1 += bool(docs and docs[0] == doc)
        hits3 += doc in docs
    blocked = 0
    for s in SQL_ATTACKS:
        try:
            tools.query_sales_sql(s)
        except ValueError:
            blocked += 1
    agent = Agent()
    routed = sum(agent.run(q)["tool_calls"][0]["tool"] == t for q, t in AGENT_CASES)
    m = predict.dimensions()["metrics"]
    res = {
        "retrieval_hit@1": hits1 / len(RETRIEVAL), "retrieval_hit@3": hits3 / len(RETRIEVAL),
        "forecast_wmape": round(m["val_wmape"], 4), "baseline_wmape": round(m["baseline_wmape_roll28"], 4),
        "forecast_improvement_vs_baseline": round(m["improvement_vs_baseline"], 3),
        "sql_attacks_blocked": f"{blocked}/{len(SQL_ATTACKS)}",
        "agent_routing_accuracy": routed / len(AGENT_CASES), "agent_mode": agent.run("hi")["mode"],
    }
    Path(__file__).with_name("results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
