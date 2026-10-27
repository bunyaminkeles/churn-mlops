"""DataSplitter birim testleri."""

import pandas as pd
import pytest

from churn_mlops.config import DataConfig, SplitConfig
from churn_mlops.data import DataSplitter


@pytest.fixture
def labeled_df() -> pd.DataFrame:
    """%20 pozitif sınıf içeren, satırları takip edilebilir 100 satırlık veri."""
    n_rows = 100
    return pd.DataFrame(
        {
            "row_id": range(n_rows),
            "tenure": range(n_rows),
            "Churn": [1] * 20 + [0] * 80,
        }
    )


def make_split_config(**overrides: object) -> SplitConfig:
    """Varsayılan ayrım ayarlarını, verilen değerlerle güncelleyerek oluşturur."""
    params = {"test_size": 0.2, "random_state": 42, "stratify": True}
    params.update(overrides)
    return SplitConfig(**params)


def test_split_returns_expected_sizes(
    data_config: DataConfig, labeled_df: pd.DataFrame
) -> None:
    train_df, test_df = DataSplitter(data_config, make_split_config()).split(labeled_df)

    assert len(train_df) == 80
    assert len(test_df) == 20


def test_split_has_no_overlap_and_no_data_loss(
    data_config: DataConfig, labeled_df: pd.DataFrame
) -> None:
    train_df, test_df = DataSplitter(data_config, make_split_config()).split(labeled_df)

    train_ids = set(train_df["row_id"])
    test_ids = set(test_df["row_id"])
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(labeled_df["row_id"])


def test_split_preserves_class_ratio_when_stratified(
    data_config: DataConfig, labeled_df: pd.DataFrame
) -> None:
    train_df, test_df = DataSplitter(data_config, make_split_config()).split(labeled_df)

    assert train_df["Churn"].mean() == pytest.approx(0.2)
    assert test_df["Churn"].mean() == pytest.approx(0.2)


def test_split_is_reproducible_with_same_seed(
    data_config: DataConfig, labeled_df: pd.DataFrame
) -> None:
    splitter = DataSplitter(data_config, make_split_config())

    first_train, first_test = splitter.split(labeled_df)
    second_train, second_test = splitter.split(labeled_df)

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)


def test_split_differs_with_different_seed(
    data_config: DataConfig, labeled_df: pd.DataFrame
) -> None:
    _, test_a = DataSplitter(data_config, make_split_config(random_state=1)).split(
        labeled_df
    )
    _, test_b = DataSplitter(data_config, make_split_config(random_state=2)).split(
        labeled_df
    )

    assert set(test_a["row_id"]) != set(test_b["row_id"])
