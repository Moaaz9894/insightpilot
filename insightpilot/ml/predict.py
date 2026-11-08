from __future__ import annotations
from functools import lru_cache
import duckdb
import joblib
import pandas as pd
from ..config import db_path, model_path
from .features import FEATURES, build_features


@lru_cache(maxsize=1)
def _bundle():
    return joblib.load(model_path())


def reset_cache():
    _bundle.cache_clear()


def dimensions() -> dict:
    b = _bundle()
    return {"skus": b["skus"], "regions": b["regions"], "metrics": b["metrics"]}


def forecast(sku: str, region: str, horizon_days: int = 14, promo: bool = False) -> dict:
    b = _bundle()
    sku, region = sku.upper(), region.capitalize()
    if sku not in b["skus"]:
        raise ValueError(f"Unknown sku {sku}. Valid: {b['skus']}")
    if region not in b["regions"]:
        raise ValueError(f"Unknown region {region}. Valid: {b['regions']}")
    if not 1 <= horizon_days <= 28:
        raise ValueError("horizon_days must be between 1 and 28")

    con = duckdb.connect(str(db_path()), read_only=True)
    hist = con.execute(
        """SELECT * FROM gold_sku_region_daily WHERE sku = ? AND region = ?
           ORDER BY sale_date DESC LIMIT 120""", [sku, region]).df().sort_values("sale_date")
    con.close()
    hist["sale_date"] = pd.to_datetime(hist["sale_date"])
    last = hist["sale_date"].max()
    base_price = float(hist.loc[hist["promo_flag"] == 0, "avg_price"].tail(28).mean())
    fut = pd.DataFrame({
        "sale_date": pd.date_range(last + pd.Timedelta(days=1), periods=horizon_days),
        "sku": sku, "region": region, "units": float("nan"),
        "avg_price": base_price * (0.85 if promo else 1.0), "promo_flag": int(promo)})
    feats = build_features(pd.concat([hist, fut], ignore_index=True), b["skus"], b["regions"])
    feats = feats[feats["sale_date"] > last]
    yhat = b["model"].predict(feats[FEATURES]).clip(min=0)
    rows = [{"date": d.strftime("%Y-%m-%d"), "forecast_units": round(float(v), 1)}
            for d, v in zip(feats["sale_date"], yhat)]
    return {"sku": sku, "region": region, "promo_assumed": promo,
            "horizon_days": horizon_days, "total_units": round(float(yhat.sum()), 1),
            "daily": rows, "model_val_wmape": round(b["metrics"]["val_wmape"], 4)}
