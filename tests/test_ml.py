import joblib
import pytest
from insightpilot.config import model_path
from insightpilot.ml import predict


def test_model_beats_naive_baseline_by_a_wide_margin():
    m = joblib.load(model_path())["metrics"]
    assert m["val_wmape"] < 0.15
    assert m["val_wmape"] < 0.7 * m["baseline_wmape_roll28"]


def test_forecast_shape_and_values():
    r = predict.forecast("SKU-001", "North", 14)
    assert len(r["daily"]) == 14
    assert all(d["forecast_units"] >= 0 for d in r["daily"])


def test_promo_does_not_decrease_forecast():
    base = predict.forecast("SKU-006", "East", 14, promo=False)["total_units"]
    promo = predict.forecast("SKU-006", "East", 14, promo=True)["total_units"]
    assert promo >= base * 0.98


@pytest.mark.parametrize("sku,region,h", [("SKU-999", "North", 7), ("SKU-001", "Mars", 7), ("SKU-001", "North", 60)])
def test_forecast_rejects_bad_input(sku, region, h):
    with pytest.raises(ValueError):
        predict.forecast(sku, region, h)
