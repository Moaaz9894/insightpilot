# Databricks notebook source
# MAGIC %md
# MAGIC # InsightPilot on Databricks: Medallion + MLflow + Model Serving
# MAGIC UNTESTED on a live workspace (the local DuckDB version in `sql/` is the tested reference).
# MAGIC Upload `store/raw/sales_raw.csv` to a Unity Catalog Volume first.

# COMMAND ----------
CATALOG, SCHEMA = "main", "insightpilot"
RAW = "/Volumes/main/insightpilot/raw/sales_raw.csv"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE {CATALOG}.{SCHEMA}")

# COMMAND ----------
# MAGIC %sql
# MAGIC -- BRONZE
# MAGIC CREATE OR REPLACE TABLE bronze_sales AS
# MAGIC SELECT *, current_timestamp() AS _ingested_at
# MAGIC FROM read_files('/Volumes/main/insightpilot/raw/sales_raw.csv', format => 'csv', header => true);

# COMMAND ----------
# MAGIC %sql
# MAGIC -- SILVER
# MAGIC CREATE OR REPLACE TABLE silver_sales AS
# MAGIC SELECT CAST(sale_date AS DATE) sale_date, store_id, region, channel, sku, product_name, category,
# MAGIC        CAST(units AS INT) units, CAST(unit_price AS DOUBLE) unit_price, CAST(promo_flag AS INT) promo_flag
# MAGIC FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY sale_date, store_id, sku ORDER BY _ingested_at) rn FROM bronze_sales)
# MAGIC WHERE rn = 1 AND units IS NOT NULL AND units >= 0 AND unit_price > 0;

# COMMAND ----------
# MAGIC %sql
# MAGIC -- GOLD
# MAGIC CREATE OR REPLACE TABLE gold_daily_sales AS
# MAGIC SELECT *, units * unit_price AS revenue FROM silver_sales;
# MAGIC CREATE OR REPLACE TABLE gold_sku_region_daily AS
# MAGIC SELECT sale_date, region, sku, SUM(units) units,
# MAGIC        SUM(units*unit_price)/NULLIF(SUM(units),0) avg_price, MAX(promo_flag) promo_flag
# MAGIC FROM silver_sales GROUP BY 1,2,3;

# COMMAND ----------
# Train with the SAME feature code as the repo (pip install the repo or %run the module), then register to Unity Catalog.
import mlflow
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Shared/insightpilot-demand-forecast")
# from insightpilot.ml.features import build_features, FEATURES
# df = spark.table("gold_sku_region_daily").toPandas()  ... same training code as insightpilot/ml/train.py ...
# mlflow.lightgbm.log_model(model, name="model", registered_model_name=f"{CATALOG}.{SCHEMA}.demand_forecast")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Serve
# MAGIC Create a **Model Serving endpoint** from the registered model (UI or `databricks serving-endpoints create`).
# MAGIC Then point `get_forecast` at that endpoint (REST) instead of the local joblib bundle.
