#!/usr/bin/env bash
# Deploys the API to Azure Container Apps. UNTESTED against a live subscription - review before running.
# Prereqs: az login; an Azure OpenAI resource + deployments; (optional) an Azure AI Search service.
set -euo pipefail
RG=${RG:-insightpilot-rg}; LOC=${LOC:-westeurope}
ACR=${ACR:-insightpilotacr$RANDOM}; ENV=${ENV:-insightpilot-env}; APP=${APP:-insightpilot-api}

az group create -n "$RG" -l "$LOC"
# Budget guard FIRST: avoid surprise bills while experimenting (Cost Management > Budgets).
az acr create -g "$RG" -n "$ACR" --sku Basic
az acr build -r "$ACR" -t insightpilot:latest .
az containerapp env create -g "$RG" -n "$ENV" -l "$LOC"
az containerapp create -g "$RG" -n "$APP" --environment "$ENV" \
  --image "$ACR.azurecr.io/insightpilot:latest" --registry-server "$ACR.azurecr.io" \
  --target-port 8000 --ingress external --min-replicas 0 --max-replicas 2 \
  --secrets aoai-key="$AZURE_OPENAI_API_KEY" api-key="$API_KEY" search-key="${AZURE_SEARCH_KEY:-none}" \
  --env-vars AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" AZURE_OPENAI_DEPLOYMENT="$AZURE_OPENAI_DEPLOYMENT" \
             AZURE_OPENAI_API_KEY=secretref:aoai-key API_KEY=secretref:api-key \
             AZURE_SEARCH_ENDPOINT="${AZURE_SEARCH_ENDPOINT:-}" AZURE_SEARCH_KEY=secretref:search-key
az containerapp show -g "$RG" -n "$APP" --query properties.configuration.ingress.fqdn -o tsv
