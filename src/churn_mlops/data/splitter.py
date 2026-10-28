"""Veriyi eğitim ve test kümelerine ayırma."""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_mlops.config.settings import DataConfig, SplitConfig

logger = logging.getLogger(__name__)


class DataSplitter:
    """Temizlenmiş veriyi tekrarlanabilir şekilde eğitim ve test kümelerine ayırır."""

    def __init__(self, data_config: DataConfig, split_config: SplitConfig) -> None:
        self._data_config = data_config
        self._split_config = split_config

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Veriyi ayırır ve (train_df, test_df) olarak döndürür."""
        target = df[self._data_config.target_column]
        stratify = target if self._split_config.stratify else None

        train_df, test_df = train_test_split(
            df,
            test_size=self._split_config.test_size,
            random_state=self._split_config.random_state,
            stratify=stratify,
        )

        logger.info(
            "Split data: %d train rows, %d test rows", len(train_df), len(test_df)
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
