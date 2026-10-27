"""DataCleaner birim testleri."""

import pandas as pd
import pytest

from churn_mlops.config.settings import DataConfig
from churn_mlops.data import DataCleaner, DataValidationError


@pytest.fixture
def cleaner(data_config: DataConfig) -> DataCleaner:
    return DataCleaner(data_config)


def test_clean_preserves_row_count(
    cleaner: DataCleaner, sample_raw_df: pd.DataFrame
) -> None:
    cleaned = cleaner.clean(sample_raw_df)

    assert len(cleaned) == len(sample_raw_df)


def test_clean_drops_id_column(
    cleaner: DataCleaner, sample_raw_df: pd.DataFrame
) -> None:
    cleaned = cleaner.clean(sample_raw_df)

    assert "customerID" not in cleaned.columns


def test_clean_fills_blank_total_charges_for_new_customers(
    cleaner: DataCleaner, sample_raw_df: pd.DataFrame
) -> None:
    cleaned = cleaner.clean(sample_raw_df)

    assert cleaned["TotalCharges"].dtype == "float64"
    assert cleaned.loc[2, "TotalCharges"] == 0.0
    assert cleaned.loc[1, "TotalCharges"] == 1889.5


def test_clean_encodes_target_as_binary(
    cleaner: DataCleaner, sample_raw_df: pd.DataFrame
) -> None:
    cleaned = cleaner.clean(sample_raw_df)

    assert cleaned["Churn"].tolist() == [0, 0, 1]


def test_clean_does_not_mutate_input(
    cleaner: DataCleaner, sample_raw_df: pd.DataFrame
) -> None:
    original = sample_raw_df.copy()

    cleaner.clean(sample_raw_df)

    pd.testing.assert_frame_equal(sample_raw_df, original)


def test_clean_raises_on_missing_total_charges_for_existing_customer(
    cleaner: DataCleaner, sample_raw_df: pd.DataFrame
) -> None:
    sample_raw_df.loc[1, "TotalCharges"] = " "

    with pytest.raises(DataValidationError, match="tenure > 0"):
        cleaner.clean(sample_raw_df)


@pytest.mark.parametrize("bad_label", ["yes", "Y", "Unknown"])
def test_clean_raises_on_unknown_target_label(
    cleaner: DataCleaner, sample_raw_df: pd.DataFrame, bad_label: str
) -> None:
    sample_raw_df.loc[0, "Churn"] = bad_label

    with pytest.raises(DataValidationError, match="Unexpected values"):
        cleaner.clean(sample_raw_df)
