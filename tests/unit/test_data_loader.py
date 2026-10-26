"""DataLoader birim testleri."""

import pandas as pd
import pytest

from churn_mlops.config.settings import DataConfig
from churn_mlops.data import DataLoader, DataValidationError


def test_load_returns_dataframe(
    sample_raw_df: pd.DataFrame, data_config: DataConfig
) -> None:
    sample_raw_df.to_csv(data_config.raw_path, index=False)

    df = DataLoader(data_config).load()

    assert df.shape == sample_raw_df.shape
    assert list(df.columns) == list(sample_raw_df.columns)


def test_load_preserves_raw_values(
    sample_raw_df: pd.DataFrame, data_config: DataConfig
) -> None:
    sample_raw_df.to_csv(data_config.raw_path, index=False)

    df = DataLoader(data_config).load()

    assert df.loc[2, "TotalCharges"] == " "


def test_load_raises_when_file_missing(data_config: DataConfig) -> None:
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        DataLoader(data_config).load()


def test_load_raises_on_empty_data(
    sample_raw_df: pd.DataFrame, data_config: DataConfig
) -> None:
    sample_raw_df.iloc[0:0].to_csv(data_config.raw_path, index=False)

    with pytest.raises(DataValidationError, match="empty"):
        DataLoader(data_config).load()


@pytest.mark.parametrize("column", ["customerID", "Churn"])
def test_load_raises_when_required_column_missing(
    sample_raw_df: pd.DataFrame, data_config: DataConfig, column: str
) -> None:
    sample_raw_df.drop(columns=column).to_csv(data_config.raw_path, index=False)

    with pytest.raises(DataValidationError, match=column):
        DataLoader(data_config).load()


def test_load_raises_on_duplicate_ids(
    sample_raw_df: pd.DataFrame, data_config: DataConfig
) -> None:
    duplicated_df = pd.concat([sample_raw_df, sample_raw_df.iloc[[0]]])
    duplicated_df.to_csv(data_config.raw_path, index=False)

    with pytest.raises(DataValidationError, match="duplicate"):
        DataLoader(data_config).load()
