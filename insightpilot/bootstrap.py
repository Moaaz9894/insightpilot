"""Idempotent: make sure data, warehouse and model exist. Used by Docker + tests.
  python -m insightpilot.bootstrap            # build whatever is missing
  python -m insightpilot.bootstrap --rebuild  # rebuild warehouse + model from the existing raw CSV"""
from __future__ import annotations
import argparse
from . import config
from .data.generate import generate
from .data.pipeline import run as run_pipeline
from .ml.train import train


def ensure(rebuild: bool = False) -> None:
    if rebuild:
        for p in (config.db_path(), config.model_path()):
            p.unlink(missing_ok=True)
    if not config.raw_csv().exists():
        print("no raw data found -> generating SYNTHETIC data:", generate())
    if not config.db_path().exists():
        print("pipeline ->", run_pipeline())
    if not config.model_path().exists():
        print("training ->", train())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ensure(ap.parse_args().rebuild)
