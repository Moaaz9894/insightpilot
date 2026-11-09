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
    return {"skus": b["skus"], "regions": b["regions"], "metrics": b["metrics"],
            "data_end": b.get("data_end")}


def _resolve(value: str, options: list[str], what: str) -> str:
    """Case-insensitive match to the canonical name (handles 'Santo Domingo de los Tsachilas')."""
    m = {o.lower(): o for o in options}
    key = str(value).strip().lower()
    if key not in m:
        raise ValueError(f"Unknown {what} '{value}'. Valid: {options}")
    return m[key]


def forecast(sku: str, region: str, horizon_days: int = 14, promo: bool = False) -> dict:
    """promo=False: promotion intensity stays at its recent average. promo=True: assume a heavy
    promotion (90th percentile of the series' recent promo_items)."""
    b = _bundle()
    sku, region = _resolve(sku, b["skus"], "sku"), _resolve(region, b["regions"], "region")
    if not 1 <= horizon_days <= 28:
        raise ValueError("horizon_days must be between 1 and 28")

    con = duckdb.connect(str(db_path()), read_only=True)
    hist = con.execute(
        """SELECT * FROM gold_sku_region_daily WHERE sku = ? AND region = ?
           ORDER BY sale_date DESC LIMIT 120""", [sku, region]).df().sort_values("sale_date")
    con.close()
    hist["sale_date"] = pd.to_datetime(hist["sale_date"])
    last = hist["sale_date"].max()
    recent_promo = float(hist["promo_items"].tail(28).mean())
    promo_val = max(float(hist["promo_items"].quantile(0.9)), recent_promo) if promo else recent_promo
    oil = float(hist["oil_price"].dropna().iloc[-1]) if hist["oil_price"].notna().any() else float("nan")
    fut = pd.DataFrame({
        "sale_date": pd.date_range(last + pd.Timedelta(days=1), periods=horizon_days),
        "sku": sku, "region": region, "units": float("nan"),
        "promo_items": promo_val, "is_holiday": 0, "oil_price": oil})
    feats = build_features(pd.concat([hist, fut], ignore_index=True), b["skus"], b["regions"])
    feats = feats[feats["sale_date"] > last]
    yhat = b["model"].predict(feats[FEATURES]).clip(min=0)
    rows = [{"date": d.strftime("%Y-%m-%d"), "forecast_units": round(float(v), 1)}
            for d, v in zip(feats["sale_date"], yhat)]
    return {"sku": sku, "region": region, "promo_assumed": promo, "horizon_days": horizon_days,
            "total_units": round(float(yhat.sum()), 1), "daily": rows,
            "history_ends": last.strftime("%Y-%m-%d"), "model_val_wmape": round(b["metrics"]["val_wmape"], 4)}
