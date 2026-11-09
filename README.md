# InsightPilot — AI Copilot for Sales & Supply-Chain Teams

An end-to-end AI system that lets business users ask questions in plain language and get answers that are
**grounded in data, forecasts and company policy**: no hallucinated numbers.

> "Forecast BEVERAGES in Guayas for the next 14 days" → ML demand forecast
> "Top categories by units sold?" → governed SQL on the warehouse
> "What's the max promo discount?" → retrieved policy, cited

## Business problem
Commercial teams (Route-to-Market, Revenue Growth Management, Supply Chain, Finance) wait on analysts for
routine questions: past performance, expected demand, and "what are the rules?". InsightPilot answers all three
in seconds through one interface, and exposes the same capabilities to other systems via REST and MCP.
Expected impact: fewer ad-hoc analyst requests, faster promo/replenishment decisions, one governed source of truth.

## Architecture
```
 raw sales ─► BRONZE ─► SILVER (clean, dedupe) ─► GOLD ──┬─► Demand model (LightGBM, MLflow) ─┐
 (CSV/ERP)     Delta/DuckDB                              └─► SQL tool (read-only, guarded)   │
 policy docs ─► chunk ─► Azure AI Search (hybrid) ─────────► Retrieval tool                   │
                                                                                              ▼
 Chat UI ─► FastAPI /chat ─► Agent (Azure OpenAI function calling) ◄── tools ──► MCP server (any MCP client)
              │ Docker ─► Azure Container Apps · CI/CD GitHub Actions · API-key auth · latency/tool logging
```
| Layer | Local (tested) | Cloud target |
|---|---|---|
| Lakehouse | DuckDB, `sql/*.sql` | Databricks Delta + Unity Catalog (`databricks/` notebook) |
| ML tracking/registry | MLflow (SQLite) | Databricks MLflow + Model Serving |
| Retrieval | TF-IDF | Azure AI Search hybrid (keyword + vector) |
| LLM | offline demo router | Azure OpenAI (function calling) |
| Hosting | uvicorn | Docker → Azure Container Apps |

## Results (real data: Favorita, 3.0M rows)
Dataset: Kaggle *Store Sales – Time Series Forecasting* (54 stores, 33 product families, 16 regions, 2013–2017).
The model forecasts 528 series (family × region) 28 days ahead and is evaluated on the **last 56 days, held out from training**.

| Metric | Value |
|---|---|
| Forecast WMAPE (LightGBM) | **11.7 %** vs 20.2 % naive baseline (**−42 %** error) |
| Retrieval hit@1 / hit@3 | 100 % / 100 % (10 questions) |
| SQL-injection/forbidden-query attempts blocked | 6 / 6 |
| Agent tool-routing accuracy (offline) | 3 / 3 |
| Automated tests | 31 passing (pipeline, ML, security, RAG, API, MCP, real-data ingest) |

![API demo](docs/img/api-chat-real.png)

Baseline = rolling 28-day mean of past demand. Features: calendar, promotion intensity, national holidays, oil price, 28-day lag/rolling demand.
**CI runs on a synthetic dataset** (no Kaggle access in CI); the real-data numbers above come from running the ingest locally (`make eval` or `python -m eval.run_eval`).

## Quick start
Synthetic demo data (no downloads needed):
```bash
make setup && make bootstrap     # generate data → Bronze/Silver/Gold → train + register model
make test && make eval
make api                         # http://localhost:8000/docs
make ui                          # chat UI
make mcp                         # MCP server (stdio)
```
Real data (Windows PowerShell; download the Favorita files from Kaggle first):
```powershell
$env:INSIGHTPILOT_HOME = "$PWD\store\favorita"
python -m insightpilot.data.ingest_favorita --src <path-to-favorita-folder>
python -m insightpilot.bootstrap
python -m eval.run_eval
uvicorn insightpilot.api.main:app --reload
```
Works offline with zero keys (demo router). Copy `.env.example` → `.env` and add Azure keys for full LLM + vector search.

## Key engineering decisions
- **No train/serve skew**: one `build_features()` for training and serving.
- **28-day lag features** → any horizon ≤ 28 days is forecast in one pass from real history (no recursive error). Lags are computed on a complete daily calendar, so closed-store days (e.g. Dec 25) never shift them.
- **Domain features:** promotion intensity, national holidays (moved holidays and make-up work days handled), oil price, pay-days.
- **Schema-checked ingest:** missing or renamed source columns fail fast with a clear message.
- **Safe text-to-SQL**: single `SELECT/WITH` only, keyword/function deny-list, **read-only connection with external access disabled**, 200-row cap, queries only the Gold layer. Defence in depth.
- **Data-quality gates** in the pipeline fail the run instead of serving bad data.
- **Tools defined once**, reused by the agent, REST API and MCP server.
- **Errors go back to the model** so it can self-correct; step limit prevents loops.
- Auth via API key (secret store in cloud), structured logs of tool calls + latency.

## Azure deployment
1. Create Azure OpenAI (chat + `text-embedding-3-small` deployments) and Azure AI Search.
2. `python -m insightpilot.rag.retriever` → indexes policies into AI Search.
3. `bash infra/deploy_azure.sh` → builds in ACR and deploys to Container Apps (scale-to-zero).
4. Set a **cost budget alert** first. Move secrets to Key Vault for production.

> **Honest status:** local stack, tests, eval and real-data (Favorita) runs are verified. `AzureSearchRetriever`,
> the Azure OpenAI loop, `deploy_azure.sh` and the Databricks notebook are written to the SDK/CLI docs but **not run
> against live cloud resources**; expect small fixes on first deploy.

## Use cases it adapts to (swap data + docs, keep the platform)
Route-to-Market (outlet/route questions) · Revenue Growth Management (promo effectiveness, price rules) ·
Supply Chain (demand-driven safety stock) · Finance (credit/returns policy Q&A, sales analytics).

## Evaluation & next steps
- Add LLM-as-judge for groundedness/answer-relevance on a golden set once Azure OpenAI is attached.
- Add prices to enable revenue and price-elasticity analysis (Favorita has units only); add hierarchical/Prophet baselines and prediction intervals.
- Move forecasting to a Databricks Model Serving endpoint; add drift monitoring and scheduled retraining job.
