"""ProjectConfig yükleme ve doğrulama testleri."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from churn_mlops.config import ProjectConfig

VALID_CONFIG = {
    "data": {
        "raw_path": "data/raw/sample.csv",
        "interim_path": "data/interim/sample_clean.csv",
        "train_path": "data/processed/train.csv",
        "test_path": "data/processed/test.csv",
        "target_column": "Churn",
        "id_column": "customerID",
    },
    "split": {
        "test_size": 0.2,
        "random_state": 42,
        "stratify": True,
    },
}


def write_yaml(path: Path, content: dict) -> Path:
    """Verilen sözlüğü YAML dosyası olarak yazar ve yolunu döndürür."""
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def test_from_yaml_loads_valid_config(tmp_path: Path) -> None:
    config_path = write_yaml(tmp_path / "params.yaml", VALID_CONFIG)

    config = ProjectConfig.from_yaml(config_path)

    assert config.data.raw_path == Path("data/raw/sample.csv")
    assert config.data.target_column == "Churn"
    assert config.data.id_column == "customerID"
    assert config.split.test_size == 0.2
    assert config.split.random_state == 42
    assert config.split.stratify is True


def test_from_yaml_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ProjectConfig.from_yaml(tmp_path / "missing.yaml")


def test_from_yaml_rejects_unknown_field(tmp_path: Path) -> None:
    invalid_config = {
        **VALID_CONFIG,
        "data": {**VALID_CONFIG["data"], "taget_column": "Churn"},
    }
    config_path = write_yaml(tmp_path / "params.yaml", invalid_config)

    with pytest.raises(ValidationError, match="taget_column"):
        ProjectConfig.from_yaml(config_path)


def test_from_yaml_rejects_missing_field(tmp_path: Path) -> None:
    data_without_target = {
        key: value
        for key, value in VALID_CONFIG["data"].items()
        if key != "target_column"
    }
    invalid_config = {**VALID_CONFIG, "data": data_without_target}
    config_path = write_yaml(tmp_path / "params.yaml", invalid_config)

    with pytest.raises(ValidationError, match="target_column"):
        ProjectConfig.from_yaml(config_path)


@pytest.mark.parametrize("test_size", [0, 1, 1.5, -0.1])
def test_from_yaml_rejects_invalid_test_size(tmp_path: Path, test_size: float) -> None:
    invalid_config = {
        **VALID_CONFIG,
        "split": {**VALID_CONFIG["split"], "test_size": test_size},
    }
    config_path = write_yaml(tmp_path / "params.yaml", invalid_config)

    with pytest.raises(ValidationError, match="test_size"):
        ProjectConfig.from_yaml(config_path)


def test_config_is_immutable(tmp_path: Path) -> None:
    config = ProjectConfig.from_yaml(write_yaml(tmp_path / "params.yaml", VALID_CONFIG))

    with pytest.raises(ValidationError):
        config.data.target_column = "Other"


def test_project_params_file_is_valid() -> None:
    config = ProjectConfig.from_yaml(Path("params.yaml"))

    assert config.data.raw_path.name == "telco_churn.csv"
