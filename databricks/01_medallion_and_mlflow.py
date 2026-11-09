# Databricks notebook source
# MAGIC %md
# MAGIC # InsightPilot on Databricks
# MAGIC **Medallion (Delta) → MLflow training + Unity Catalog registry → batch forecasts table.**
# MAGIC
# MAGIC Setup (once): *Workspace → Create → Git folder* → paste the repo URL, then open this notebook from the
# MAGIC cloned folder and attach **Serverless** compute. Run the cells in order.
# MAGIC
# MAGIC The Silver/Gold SQL and the feature/training code are the **same files** the local project uses
# MAGIC (`sql/*.sql`, `insightpilot/ml/*`), so local and cloud results are directly comparable.

# COMMAND ----------
# MAGIC %pip install lightgbm

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import os, sys

# ---- configuration (change CATALOG if yours is not "workspace") ----
CATALOG, SCHEMA, VOLUME = "workspace", "insightpilot", "raw"
RAW_FILE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/sales_raw.csv"

# find the repo root (this notebook lives in <repo>/databricks/)
root = os.getcwd()
while root != "/" and not os.path.isdir(os.path.join(root, "insightpilot", "ml")):
    root = os.path.dirname(root)
assert root != "/", "Open this notebook from inside the cloned Git folder"
REPO_ROOT = root
sys.path.insert(0, REPO_ROOT)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
spark.sql(f"USE {CATALOG}.{SCHEMA}")
print("repo:", REPO_ROOT, "| volume ready:", f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Upload the raw data (manual, once)
# MAGIC Catalog → **workspace → insightpilot → raw (volume) → Upload to this volume** → choose
# MAGIC `store\favorita\raw\sales_raw.csv` from your machine. Then run the next cell to verify.

# COMMAND ----------
files = [f.name for f in dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/")]
assert "sales_raw.csv" in files, f"Upload sales_raw.csv to the volume first. Found: {files}"
print("found sales_raw.csv")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Bronze → Silver → Gold (Delta tables)

# COMMAND ----------
# BRONZE: raw as received + ingestion timestamp (Spark flavour of sql/01_bronze.sql)
spark.sql(f"""
CREATE OR REPLACE TABLE bronze_sales AS
SELECT *, current_timestamp() AS _ingested_at
FROM read_files('{RAW_FILE}', format => 'csv', header => true, inferSchema => true)
""")

# SILVER + GOLD: run the project's own SQL files unchanged
for name in ("02_silver.sql", "03_gold.sql"):
    sql_text = open(os.path.join(REPO_ROOT, "sql", name)).read()
    for stmt in filter(str.strip, sql_text.split(";")):
        spark.sql(stmt)

# COMMAND ----------
# data-quality gates (same rules as insightpilot/data/pipeline.py)
counts = {t: spark.table(t).count() for t in
          ("bronze_sales", "silver_sales", "gold_daily_sales", "gold_monthly_summary", "gold_sku_region_daily")}
print(counts)
assert counts["silver_sales"] <= counts["bronze_sales"]
assert spark.sql("SELECT COUNT(*) c FROM silver_sales WHERE units < 0").first()["c"] == 0
dupes = spark.sql("""SELECT COUNT(*) c FROM (SELECT 1 FROM silver_sales
                     GROUP BY sale_date, store_id, sku HAVING COUNT(*) > 1)""").first()["c"]
assert dupes == 0, "duplicates survived Silver"

# COMMAND ----------
# MAGIC %sql
# MAGIC -- quick business view straight from Gold
# MAGIC SELECT category, SUM(units) AS units FROM gold_daily_sales GROUP BY category ORDER BY units DESC

# COMMAND ----------
# MAGIC %md
# MAGIC ## Train + track + register (MLflow / Unity Catalog)

# COMMAND ----------
import mlflow
import mlflow.lightgbm
from mlflow.models import infer_signature
from lightgbm import LGBMRegressor
from insightpilot.ml.features import FEATURES
from insightpilot.ml.train import PARAMS, VAL_DAYS, fit_and_evaluate

pdf = spark.table("gold_sku_region_daily").toPandas()
res = fit_and_evaluate(pdf)           # hold out the last 56 days, same as local
print(res["metrics"])

# COMMAND ----------
mlflow.set_registry_uri("databricks-uc")
MODEL_NAME = f"{CATALOG}.{SCHEMA}.demand_forecast"

feat = res["feat"]
with mlflow.start_run(run_name="lightgbm-databricks") as run:
    mlflow.log_params({**PARAMS, "val_days": VAL_DAYS, "features": ",".join(FEATURES)})
    mlflow.log_metrics(res["metrics"])
    final = LGBMRegressor(**PARAMS).fit(feat[FEATURES], feat["units"])      # refit on all data for serving
    X = feat[FEATURES].astype("float64")
    mlflow.lightgbm.log_model(final, artifact_path="model", registered_model_name=MODEL_NAME,
                              signature=infer_signature(X, final.predict(X)), input_example=X.head(5))
print("registered:", MODEL_NAME, "| run:", run.info.run_id)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Batch-score every series (28 days ahead) into a Delta table

# COMMAND ----------
from insightpilot.ml.batch import batch_forecast

hist = spark.sql("""SELECT * FROM gold_sku_region_daily
                    WHERE sale_date >= (SELECT date_sub(MAX(sale_date), 150) FROM gold_sku_region_daily)""").toPandas()
fc = batch_forecast(hist, final, res["skus"], res["regions"], horizon=28)
spark.createDataFrame(fc).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("gold_forecast_28d")
print(len(fc), "forecast rows written to gold_forecast_28d")

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT region, sku, ROUND(SUM(forecast_units)) AS units_next_28d
# MAGIC FROM gold_forecast_28d GROUP BY region, sku ORDER BY units_next_28d DESC LIMIT 10

# COMMAND ----------
# MAGIC %md
# MAGIC ## Serving (optional)
# MAGIC If your workspace offers **Model Serving**, create an endpoint from the registered model
# MAGIC (*Serving → Create endpoint → select `demand_forecast`*) and point `get_forecast` at it.
# MAGIC Otherwise the `gold_forecast_28d` table is the integration point (query it via Databricks SQL).
