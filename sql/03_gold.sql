-- GOLD: business-ready tables. These are the only tables the AI agent may query.
CREATE OR REPLACE TABLE gold_daily_sales AS
SELECT sale_date, store_id, region, channel, sku, product_name, category,
       units, unit_price, units * unit_price AS revenue, promo_flag
FROM silver_sales;

CREATE OR REPLACE TABLE gold_monthly_summary AS
SELECT date_trunc('month', sale_date) AS month, region, category,
       SUM(units) AS units, ROUND(SUM(units * unit_price), 2) AS revenue,
       ROUND(AVG(promo_flag), 4) AS promo_share
FROM silver_sales
GROUP BY 1, 2, 3;

-- Training/serving table for the forecasting model (sku x region x day).
CREATE OR REPLACE TABLE gold_sku_region_daily AS
SELECT sale_date, region, sku,
       SUM(units) AS units,
       SUM(units * unit_price) / NULLIF(SUM(units), 0) AS avg_price,
       MAX(promo_flag) AS promo_flag
FROM silver_sales
GROUP BY 1, 2, 3;
