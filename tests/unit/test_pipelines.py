"""Pipeline aşamalarının uçtan uca testleri."""

import pandas as pd
import pytest

from churn_mlops.config import DataConfig, ProjectConfig, SplitConfig
from churn_mlops.pipelines import clean, split


@pytest.fixture
def project_config(data_config: DataConfig) -> ProjectConfig:
    """Geçici klasörlere işaret eden, tam bir proje konfigürasyonu."""
    return ProjectConfig(
        data=data_config,
        split=SplitConfig(test_size=0.25, random_state=42, stratify=False),
    )


def test_clean_stage_writes_cleaned_data(
    project_config: ProjectConfig, sample_raw_df: pd.DataFrame
) -> None:
    sample_raw_df.to_csv(project_config.data.raw_path, index=False)

    clean.run(project_config)

    cleaned = pd.read_csv(project_config.data.interim_path)
    assert len(cleaned) == len(sample_raw_df)
    assert "customerID" not in cleaned.columns
    assert set(cleaned["Churn"]) <= {0, 1}


def test_split_stage_writes_train_and_test(
    project_config: ProjectConfig, sample_raw_df: pd.DataFrame
) -> None:
    sample_raw_df.to_csv(project_config.data.raw_path, index=False)
    clean.run(project_config)

    split.run(project_config)

    train_df = pd.read_csv(project_config.data.train_path)
    test_df = pd.read_csv(project_config.data.test_path)
    assert len(train_df) + len(test_df) == len(sample_raw_df)
    assert len(test_df) == 1


def test_main_entry_points_run_end_to_end(
    project_config: ProjectConfig,
    sample_raw_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_raw_df.to_csv(project_config.data.raw_path, index=False)
    monkeypatch.setattr(ProjectConfig, "from_yaml", lambda: project_config)

    clean.main()
    split.main()

    assert project_config.data.train_path.exists()
    assert project_config.data.test_path.exists()
