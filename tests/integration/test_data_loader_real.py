"""Gerçek ham veri üzerinde DataLoader entegrasyon testi."""

import pytest

from churn_mlops.config import ProjectConfig
from churn_mlops.data import DataLoader

CONFIG = ProjectConfig.from_yaml()


@pytest.mark.integration
@pytest.mark.skipif(
    not CONFIG.data.raw_path.exists(),
    reason="Raw data not available. Run 'dvc pull' first.",
)
def test_real_raw_data_loads() -> None:
    df = DataLoader(CONFIG.data).load()

    assert len(df) > 0
    assert CONFIG.data.target_column in df.columns
