"""The three tools the agent (and the MCP server) can use. Single source of truth."""
from __future__ import annotations
import re
import duckdb
from ..config import db_path
from ..ml import predict
from ..rag.retriever import get_retriever

MAX_ROWS = 200



def get_schema() -> str:
    """Schema + the actual dimension values in the loaded dataset."""
    con = duckdb.connect(str(db_path()), read_only=True)
    try:
        def vals(col):
            return [r[0] for r in con.execute(f"SELECT DISTINCT {col} FROM gold_daily_sales ORDER BY 1").fetchall()]
        lo, hi = con.execute("SELECT MIN(sale_date), MAX(sale_date) FROM gold_daily_sales").fetchone()
        regions, channels, cats, skus = vals("region"), vals("channel"), vals("category"), vals("sku")
    finally:
        con.close()
    return (
        "Tables (DuckDB SQL, read-only):\n"
        "gold_daily_sales(sale_date DATE, store_id, region, channel, sku, product_name, category, "
        "units INT, promo_items INT, is_holiday INT, oil_price DOUBLE)\n"
        "gold_monthly_summary(month DATE, region, category, units, promo_share)\n"
        "There is NO price/revenue column: measure sales in units.\n"
        f"Date range: {lo} to {hi}.\n"
        f"Regions: {regions}\nChannels: {channels}\nCategories: {cats}\nSkus: {skus}")


_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|pragma|install|load|export|"
    r"import|call|truncate|vacuum|checkpoint)\b|read_\w+\s*\(|glob\s*\(|--|/\*|\bhttps?:", re.I)


def validate_sql(sql: str) -> str:
    """Defence in depth: validation here + a read-only connection with external access disabled."""
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        raise ValueError("Only a single statement is allowed")
    if not re.match(r"^(select|with)\b", s, re.I):
        raise ValueError("Only SELECT queries are allowed")
    if _FORBIDDEN.search(s):
        raise ValueError("Query contains a forbidden keyword")
    return s


def query_sales_sql(sql: str) -> dict:
    s = validate_sql(sql)
    con = duckdb.connect(str(db_path()), read_only=True, config={"enable_external_access": False})
    try:
        cur = con.execute(f"SELECT * FROM ({s}) AS q LIMIT {MAX_ROWS}")
        cols = [d[0] for d in cur.description]
        rows = [[v.isoformat() if hasattr(v, "isoformat") else v for v in r] for r in cur.fetchall()]
    finally:
        con.close()
    return {"columns": cols, "rows": rows, "row_count": len(rows), "truncated": len(rows) == MAX_ROWS}


def get_forecast(sku: str, region: str, horizon_days: int = 14, promo: bool = False) -> dict:
    return predict.forecast(sku, region, horizon_days, promo)


def search_policies(query: str, k: int = 3) -> dict:
    return {"results": get_retriever().search(query, k)}


TOOL_FUNCS = {"query_sales_sql": query_sales_sql, "get_forecast": get_forecast,
              "search_policies": search_policies, "get_schema": get_schema}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "query_sales_sql",
        "description": "Run a read-only SELECT on the sales warehouse (call get_schema first if unsure). "
                       "Use for historical facts: units sold, rankings, trends, promotions.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}},
    {"type": "function", "function": {
        "name": "get_forecast",
        "description": "ML demand forecast (units per day) for one sku in one region, up to 28 days ahead.",
        "parameters": {"type": "object", "properties": {
            "sku": {"type": "string", "description": "a value from the Skus list in the schema"},
            "region": {"type": "string", "description": "a value from the Regions list in the schema"},
            "horizon_days": {"type": "integer", "minimum": 1, "maximum": 28},
            "promo": {"type": "boolean", "description": "assume a promotion runs in the period"}},
            "required": ["sku", "region"]}}},
    {"type": "function", "function": {
        "name": "search_policies",
        "description": "Search company policy documents (pricing, promotions, distribution, supply chain, "
                       "credit & returns). Use for any 'what is the rule/limit/process' question.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_schema", "description": "Return the warehouse schema.",
        "parameters": {"type": "object", "properties": {}}}},
]
