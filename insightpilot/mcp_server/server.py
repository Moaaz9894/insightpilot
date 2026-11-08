"""MCP server exposing the same tools to ANY MCP client (Claude Desktop, VS Code, agents...).
Run:  python -m insightpilot.mcp_server.server          (stdio)
Claude Desktop config:  {"mcpServers": {"insightpilot": {"command": "python",
                         "args": ["-m", "insightpilot.mcp_server.server"]}}}"""
from mcp.server.fastmcp import FastMCP
from ..agent import tools
from ..bootstrap import ensure

mcp = FastMCP("insightpilot")


@mcp.tool()
def get_schema() -> str:
    """Return the sales warehouse schema. Call before writing SQL."""
    return tools.get_schema()


@mcp.tool()
def query_sales_sql(sql: str) -> dict:
    """Run a read-only SELECT on the sales warehouse (max 200 rows)."""
    return tools.query_sales_sql(sql)


@mcp.tool()
def get_forecast(sku: str, region: str, horizon_days: int = 14, promo: bool = False) -> dict:
    """ML demand forecast (units/day) for a SKU and region, up to 28 days ahead."""
    return tools.get_forecast(sku, region, horizon_days, promo)


@mcp.tool()
def search_policies(query: str, k: int = 3) -> dict:
    """Search company policy documents (pricing, promotions, distribution, supply chain, credit)."""
    return tools.search_policies(query, k)


if __name__ == "__main__":
    ensure()
    mcp.run()
