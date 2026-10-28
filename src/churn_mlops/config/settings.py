"""Proje konfigürasyonunun yüklenmesi ve doğrulanması."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_PARAMS_PATH = Path("params.yaml")


class DataConfig(BaseModel):
    """Veri kaynakları ve çıktı yollarıyla ilgili ayarlar."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_path: Path
    interim_path: Path
    train_path: Path
    test_path: Path
    target_column: str
    id_column: str


class SplitConfig(BaseModel):
    """Eğitim/test ayrımıyla ilgili ayarlar."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    test_size: float = Field(gt=0, lt=1)
    random_state: int
    stratify: bool


class ProjectConfig(BaseModel):
    """Projenin tüm konfigürasyonunu temsil eden kök model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data: DataConfig
    split: SplitConfig

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_PARAMS_PATH) -> "ProjectConfig":
        """YAML dosyasını okuyup doğrulanmış bir konfigürasyon nesnesi döndürür."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open(encoding="utf-8") as file:
            raw_config = yaml.safe_load(file) or {}

        return cls.model_validate(raw_config)
