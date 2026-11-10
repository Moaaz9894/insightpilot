"""Central configuration. Everything is driven by environment variables so the
same code runs locally, in Docker and on Azure."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = REPO_ROOT / "docs" / "policies"
SQL_DIR = REPO_ROOT / "sql"


def home() -> Path:
    p = Path(os.getenv("INSIGHTPILOT_HOME", REPO_ROOT / "store"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw_csv() -> Path:
    p = home() / "raw"
    p.mkdir(exist_ok=True)
    return p / "sales_raw.csv"


def db_path() -> Path:
    return home() / "warehouse.duckdb"


def model_path() -> Path:
    p = home() / "models"
    p.mkdir(exist_ok=True)
    return p / "forecast.joblib"


def mlflow_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{home() / 'mlflow.db'}")


def azure_openai_configured() -> bool:
    return all(os.getenv(k) for k in
               ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"))


def azure_search_configured() -> bool:
    return bool(os.getenv("AZURE_SEARCH_ENDPOINT") and os.getenv("AZURE_SEARCH_KEY")
                and azure_openai_configured())
