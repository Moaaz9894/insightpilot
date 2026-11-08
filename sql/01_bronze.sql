-- BRONZE: raw data exactly as received, plus ingestion metadata.
CREATE OR REPLACE TABLE bronze_sales AS
SELECT *, current_timestamp AS _ingested_at
FROM read_csv_auto('{raw_csv}', header = true);
