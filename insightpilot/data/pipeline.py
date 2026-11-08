"""Bronze -> Silver -> Gold pipeline (DuckDB locally; same SQL shape as Databricks)."""
from __future__ import annotations
import duckdb
from ..config import SQL_DIR, db_path, raw_csv


def run() -> dict:
    con = duckdb.connect(str(db_path()))
    for f in sorted(SQL_DIR.glob("0*.sql")):
        sql = f.read_text().replace("{raw_csv}", str(raw_csv()).replace("\\", "/"))
        for stmt in filter(str.strip, sql.split(";")):
            con.execute(stmt)
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("bronze_sales", "silver_sales", "gold_daily_sales",
                        "gold_monthly_summary", "gold_sku_region_daily")}
    # data-quality gates: fail the pipeline instead of serving bad data
    assert counts["silver_sales"] < counts["bronze_sales"], "Silver should drop dirty rows"
    assert con.execute("SELECT COUNT(*) FROM silver_sales WHERE units < 0").fetchone()[0] == 0
    dupes = con.execute("""SELECT COUNT(*) FROM (SELECT 1 FROM silver_sales
                           GROUP BY sale_date, store_id, sku HAVING COUNT(*) > 1)""").fetchone()[0]
    assert dupes == 0, "Duplicates survived Silver"
    con.close()
    return counts
