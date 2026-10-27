"""Ham verinin durumsuz (stateless) temizliği."""

import logging

import pandas as pd

from churn_mlops.config.settings import DataConfig
from churn_mlops.data.exceptions import DataValidationError

logger = logging.getLogger(__name__)

TOTAL_CHARGES_COLUMN = "TotalCharges"
TENURE_COLUMN = "tenure"
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"


class DataCleaner:
    """Ham veriye satır bazlı, durumsuz temizlik adımlarını uygular."""

    def __init__(self, config: DataConfig) -> None:
        self._config = config

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Temizlik adımlarını sırayla uygular ve yeni bir DataFrame döndürür."""
        cleaned = df.copy()
        cleaned = self._fix_total_charges(cleaned)
        cleaned = self._encode_target(cleaned)
        cleaned = self._drop_id_column(cleaned)
        logger.info(
            "Cleaned data: %d rows, %d columns", cleaned.shape[0], cleaned.shape[1]
        )
        return cleaned

    @staticmethod
    def _fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
        """Henüz faturalandırılmamış müşterilerin boş TotalCharges değerini 0 yapar."""
        total_charges = pd.to_numeric(df[TOTAL_CHARGES_COLUMN], errors="coerce")
        is_new_customer = df[TENURE_COLUMN] == 0

        unexpected_missing = total_charges.isna() & ~is_new_customer
        if unexpected_missing.any():
            raise DataValidationError(
                f"{unexpected_missing.sum()} rows have non-numeric "
                f"'{TOTAL_CHARGES_COLUMN}' with tenure > 0."
            )

        df[TOTAL_CHARGES_COLUMN] = total_charges.fillna(0.0)
        return df

    def _encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hedef sütunu 'Yes'/'No' değerlerinden 1/0'a dönüştürür."""
        target = self._config.target_column
        unknown_labels = set(df[target].unique()) - {POSITIVE_LABEL, NEGATIVE_LABEL}
        if unknown_labels:
            raise DataValidationError(
                f"Unexpected values in '{target}': {sorted(unknown_labels)}"
            )

        df[target] = (df[target] == POSITIVE_LABEL).astype(int)
        return df

    def _drop_id_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Model için bilgi taşımayan müşteri kimliği sütununu çıkarır."""
        return df.drop(columns=self._config.id_column)
