"""ProjectConfig yükleme ve doğrulama testleri."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from churn_mlops.config import ProjectConfig

VALID_CONFIG = {
    "data": {
        "raw_path": "data/raw/sample.csv",
        "target_column": "Churn",
        "id_column": "customerID",
    }
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


def test_from_yaml_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ProjectConfig.from_yaml(tmp_path / "missing.yaml")


def test_from_yaml_rejects_unknown_field(tmp_path: Path) -> None:
    invalid_config = {"data": {**VALID_CONFIG["data"], "taget_column": "Churn"}}
    config_path = write_yaml(tmp_path / "params.yaml", invalid_config)

    with pytest.raises(ValidationError):
        ProjectConfig.from_yaml(config_path)


def test_from_yaml_rejects_missing_field(tmp_path: Path) -> None:
    data_without_target = {
        key: value
        for key, value in VALID_CONFIG["data"].items()
        if key != "target_column"
    }
    config_path = write_yaml(tmp_path / "params.yaml", {"data": data_without_target})

    with pytest.raises(ValidationError):
        ProjectConfig.from_yaml(config_path)


def test_config_is_immutable(tmp_path: Path) -> None:
    config = ProjectConfig.from_yaml(write_yaml(tmp_path / "params.yaml", VALID_CONFIG))

    with pytest.raises(ValidationError):
        config.data.target_column = "Other"


def test_project_params_file_is_valid() -> None:
    config = ProjectConfig.from_yaml(Path("params.yaml"))

    assert config.data.raw_path.name == "telco_churn.csv"
