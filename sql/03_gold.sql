-- GOLD: business-ready tables. These are the only tables the AI agent may query.
CREATE OR REPLACE TABLE gold_daily_sales AS
SELECT sale_date, store_id, region, channel, sku, product_name, category,
       units, promo_items, is_holiday, oil_price
FROM silver_sales;

CREATE OR REPLACE TABLE gold_monthly_summary AS
SELECT date_trunc('month', sale_date) AS month, region, category,
       SUM(units) AS units,
       ROUND(AVG(CASE WHEN promo_items > 0 THEN 1.0 ELSE 0.0 END), 4) AS promo_share
FROM silver_sales
GROUP BY 1, 2, 3;

-- Training/serving table for the forecasting model (sku x region x day).
-- Missing calendar days (e.g. closed stores) are intentionally NOT filled here: the feature code handles gaps.
CREATE OR REPLACE TABLE gold_sku_region_daily AS
SELECT sale_date, region, sku,
       SUM(units) AS units,
       SUM(promo_items) AS promo_items,
       MAX(is_holiday) AS is_holiday,
       AVG(oil_price) AS oil_price
FROM silver_sales
GROUP BY 1, 2, 3
