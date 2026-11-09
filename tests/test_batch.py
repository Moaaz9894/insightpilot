import duckdb
import joblib
import pandas as pd
from insightpilot.config import db_path, model_path
from insightpilot.ml import predict
from insightpilot.ml.batch import batch_forecast


def test_batch_matches_single_series_forecast():
    b = joblib.load(model_path())
    con = duckdb.connect(str(db_path()), read_only=True)
    hist = con.execute("SELECT * FROM gold_sku_region_daily").df()
    con.close()
    out = batch_forecast(hist, b["model"], b["skus"], b["regions"], horizon=14)
    assert len(out) == len(b["skus"]) * len(b["regions"]) * 14
    sku, region = b["skus"][1], b["regions"][2]
    one = predict.forecast(sku, region, 14)
    got = out[(out.sku == sku) & (out.region == region)].sort_values("forecast_date")
    pd.testing.assert_series_equal(
        got["forecast_units"].reset_index(drop=True),
        pd.Series([d["forecast_units"] for d in one["daily"]], name="forecast_units"), check_exact=False, atol=0.11)
