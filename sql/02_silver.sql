-- SILVER: typed, de-duplicated, invalid rows removed.
CREATE OR REPLACE TABLE silver_sales AS
SELECT CAST(sale_date AS DATE)   AS sale_date,
       store_id, region, channel, sku, product_name, category,
       CAST(units AS INTEGER)    AS units,
       CAST(unit_price AS DOUBLE) AS unit_price,
       CAST(promo_flag AS INTEGER) AS promo_flag
FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY sale_date, store_id, sku ORDER BY _ingested_at) AS rn
    FROM bronze_sales
) t
WHERE rn = 1 AND units IS NOT NULL AND units >= 0 AND unit_price > 0;
