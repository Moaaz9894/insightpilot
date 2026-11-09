"""Train the demand-forecast model, track it in MLflow, register it, save a bundle."""
from __future__ import annotations
import duckdb
import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from ..config import db_path, home, mlflow_uri, model_path
from .features import FEATURES, build_features

VAL_DAYS = 56
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
              subsample=0.8, colsample_bytree=0.9, random_state=42, verbose=-1)


def wmape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / max(np.abs(y).sum(), 1e-9))


def train() -> dict:
    con = duckdb.connect(str(db_path()), read_only=True)
    df = con.execute("SELECT * FROM gold_sku_region_daily").df()
    con.close()
    skus, regions = sorted(df["sku"].unique()), sorted(df["region"].unique())
    feat = build_features(df, skus, regions).dropna(subset=["lag_28", "roll_28"])
    print(f"training rows={len(feat):,} series={len(skus) * len(regions)}")
    cutoff = pd.to_datetime(feat["sale_date"]).max() - pd.Timedelta(days=VAL_DAYS)
    is_val = pd.to_datetime(feat["sale_date"]) > cutoff
    tr, va = feat[~is_val], feat[is_val]

    mlflow.set_tracking_uri(mlflow_uri())
    if mlflow.get_experiment_by_name("demand-forecast") is None:  # keep artifacts inside the store
        mlflow.create_experiment("demand-forecast", artifact_location=(home() / "mlartifacts").as_uri())
    mlflow.set_experiment("demand-forecast")
    with mlflow.start_run(run_name="lightgbm") as run:
        model = LGBMRegressor(**PARAMS).fit(tr[FEATURES], tr["units"])
        pred = model.predict(va[FEATURES])
        metrics = {
            "val_wmape": wmape(va["units"], pred),
            "baseline_wmape_roll28": wmape(va["units"], va["roll_28"]),
        }
        metrics["improvement_vs_baseline"] = 1 - metrics["val_wmape"] / metrics["baseline_wmape_roll28"]
        mlflow.log_params({**PARAMS, "val_days": VAL_DAYS, "features": ",".join(FEATURES)})
        mlflow.log_metrics(metrics)
        try:  # Registry needs a DB-backed store (sqlite locally, Databricks in the cloud)
            mlflow.lightgbm.log_model(model, name="model", registered_model_name="demand-forecast")
        except Exception as e:  # never block serving on registry problems
            print("MLflow registry skipped:", e)
        # final model: refit on ALL data for serving
        final = LGBMRegressor(**PARAMS).fit(feat[FEATURES], feat["units"])
        joblib.dump({"model": final, "skus": skus, "regions": regions, "metrics": metrics,
                     "run_id": run.info.run_id, "data_end": str(df["sale_date"].max())[:10]}, model_path())
    return metrics
