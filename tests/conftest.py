"""Testler arasında paylaşılan fixture'lar."""

from pathlib import Path

import pandas as pd
import pytest

from churn_mlops.config.settings import DataConfig


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Gerçek veri yapısını taklit eden küçük bir örnek veri seti."""
    return pd.DataFrame(
        {
            "customerID": ["0001-AAAAA", "0002-BBBBB", "0003-CCCCC"],
            "tenure": [1, 24, 0],
            "MonthlyCharges": [29.85, 56.95, 20.25],
            "TotalCharges": ["29.85", "1889.5", " "],
            "Churn": ["No", "No", "Yes"],
        }
    )


@pytest.fixture
def data_config(tmp_path: Path) -> DataConfig:
    """Geçici klasördeki bir CSV dosyasına işaret eden test konfigürasyonu."""
    return DataConfig(
        raw_path=tmp_path / "raw.csv",
        target_column="Churn",
        id_column="customerID",
    )
