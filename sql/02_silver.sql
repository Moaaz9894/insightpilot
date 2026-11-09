-- SILVER: typed, de-duplicated, invalid rows removed.
CREATE OR REPLACE TABLE silver_sales AS
SELECT CAST(sale_date AS DATE)      AS sale_date,
       store_id, region, channel, sku, product_name, category,
       CAST(units AS INTEGER)       AS units,
       CAST(promo_items AS INTEGER) AS promo_items,
       CAST(is_holiday AS INTEGER)  AS is_holiday,
       CAST(oil_price AS DOUBLE)    AS oil_price
FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY sale_date, store_id, sku ORDER BY _ingested_at) AS rn
    FROM bronze_sales
) t
WHERE rn = 1 AND units IS NOT NULL AND units >= 0
