import pytest
from insightpilot.agent import tools


@pytest.mark.parametrize("sql", [
    "DROP TABLE gold_daily_sales", "DELETE FROM gold_daily_sales", "SELECT 1; DROP TABLE x",
    "SELECT * FROM read_csv('/etc/passwd')", "SELECT 1 -- comment", "COPY gold_daily_sales TO '/tmp/x.csv'",
    "ATTACH '/tmp/evil.db'", "PRAGMA database_list", "INSTALL httpfs"])
def test_dangerous_sql_blocked(sql):
    with pytest.raises(ValueError):
        tools.query_sales_sql(sql)


def test_select_works_and_is_capped():
    r = tools.query_sales_sql("SELECT * FROM gold_daily_sales")
    assert r["row_count"] == tools.MAX_ROWS and r["truncated"]


def test_cte_allowed():
    r = tools.query_sales_sql("WITH t AS (SELECT region, SUM(units) u FROM gold_daily_sales GROUP BY 1) SELECT * FROM t")
    assert r["row_count"] == 4
