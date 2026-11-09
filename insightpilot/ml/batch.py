"""Batch forecasting for ALL series at once (used by the Databricks job and tests).
Mirrors predict.forecast exactly: same features, same promo/oil assumptions."""
from __future__ import annotations
import pandas as pd
from .features import FEATURES, build_features

HISTORY_ROWS = 120


def batch_forecast(hist: pd.DataFrame, model, skus: list[str], regions: list[str], horizon: int = 28) -> pd.DataFrame:
    """hist: rows of gold_sku_region_daily (>= 120 recent days per series). Returns one row per
    series and future day: sku, region, forecast_date, forecast_units."""
    if not 1 <= horizon <= 28:
        raise ValueError("horizon must be between 1 and 28")
    hist = hist.copy()
    hist["sale_date"] = pd.to_datetime(hist["sale_date"])
    hist = (hist.sort_values("sale_date").groupby(["sku", "region"], group_keys=False)
                .tail(HISTORY_ROWS))
    last = hist["sale_date"].max()
    promo = (hist.groupby(["sku", "region"], group_keys=False).tail(28)
                 .groupby(["sku", "region"])["promo_items"].mean().rename("promo_items").reset_index())
    oil = float(hist["oil_price"].dropna().iloc[-1]) if hist["oil_price"].notna().any() else float("nan")
    days = pd.date_range(last + pd.Timedelta(days=1), periods=horizon)
    fut = promo.merge(pd.DataFrame({"sale_date": days}), how="cross")
    fut["units"], fut["is_holiday"], fut["oil_price"] = float("nan"), 0, oil
    feats = build_features(pd.concat([hist, fut], ignore_index=True), skus, regions)
    feats = feats[feats["sale_date"] > last].copy()
    feats["forecast_units"] = model.predict(feats[FEATURES]).clip(min=0).round(1)
    out = feats[["sku", "region", "sale_date", "forecast_units"]].rename(columns={"sale_date": "forecast_date"})
    out["forecast_date"] = out["forecast_date"].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)
