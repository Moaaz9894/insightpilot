import duckdb
from insightpilot.config import db_path


def test_silver_is_clean_and_smaller_than_bronze():
    con = duckdb.connect(str(db_path()), read_only=True)
    b = con.execute("SELECT COUNT(*) FROM bronze_sales").fetchone()[0]
    s = con.execute("SELECT COUNT(*) FROM silver_sales").fetchone()[0]
    assert s < b
    assert con.execute("SELECT COUNT(*) FROM silver_sales WHERE units < 0 OR units IS NULL").fetchone()[0] == 0


def test_gold_tables_exist():
    con = duckdb.connect(str(db_path()), read_only=True)
    names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"gold_daily_sales", "gold_monthly_summary", "gold_sku_region_daily"} <= names
