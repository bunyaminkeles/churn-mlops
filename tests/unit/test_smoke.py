"""Paketin doğru kurulduğunu doğrulayan smoke testleri."""

import importlib

import pytest

SUBMODULES = [
    "config",
    "data",
    "features",
    "models",
    "pipelines",
    "api",
    "monitoring",
    "utils",
]


def test_package_has_version() -> None:
    import churn_mlops

    assert churn_mlops.__version__ == "0.1.0"


@pytest.mark.parametrize("module_name", SUBMODULES)
def test_submodules_are_importable(module_name: str) -> None:
    module = importlib.import_module(f"churn_mlops.{module_name}")
    assert module is not None