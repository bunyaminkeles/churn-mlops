"""Proje konfigürasyonunun yüklenmesi ve doğrulanması."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

DEFAULT_PARAMS_PATH = Path("params.yaml")


class DataConfig(BaseModel):
    """Veri kaynağıyla ilgili ayarlar."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_path: Path
    target_column: str
    id_column: str


class ProjectConfig(BaseModel):
    """Projenin tüm konfigürasyonunu temsil eden kök model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data: DataConfig

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_PARAMS_PATH) -> "ProjectConfig":
        """YAML dosyasını okuyup doğrulanmış bir konfigürasyon nesnesi döndürür."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open(encoding="utf-8") as file:
            raw_config = yaml.safe_load(file) or {}

        return cls.model_validate(raw_config)
