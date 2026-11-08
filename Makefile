.PHONY: setup bootstrap test eval api ui mcp docker
setup:      ; pip install -r requirements-dev.txt
bootstrap:  ; python -m insightpilot.bootstrap
test:       ; python -m pytest -q tests
eval:       ; python -m eval.run_eval
api:        ; uvicorn insightpilot.api.main:app --reload
ui:         ; streamlit run ui/app.py
mcp:        ; python -m insightpilot.mcp_server.server
docker:     ; docker build -t insightpilot . && docker run -p 8000:8000 --env-file .env insightpilot
