import asyncio
from insightpilot.mcp_server.server import mcp


def test_mcp_exposes_tools_and_runs_one():
    async def go():
        names = {t.name for t in await mcp.list_tools()}
        assert names == {"get_schema", "query_sales_sql", "get_forecast", "search_policies"}
        res = await mcp.call_tool("get_forecast", {"sku": "SKU-001", "region": "North", "horizon_days": 7})
        assert res
    asyncio.run(go())
