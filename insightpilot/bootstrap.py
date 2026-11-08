"""Idempotent: make sure data, warehouse and model exist. Used by Docker + tests."""
from __future__ import annotations
from . import config
from .data.generate import generate
from .data.pipeline import run as run_pipeline
from .ml.train import train


def ensure(force: bool = False) -> None:
    if force or not config.raw_csv().exists():
        print("generating raw data ->", generate())
    if force or not config.db_path().exists():
        print("pipeline ->", run_pipeline())
    if force or not config.model_path().exists():
        print("training ->", train())


if __name__ == "__main__":
    ensure()
