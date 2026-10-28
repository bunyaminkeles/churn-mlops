"""Ham verinin okunması ve yapısal doğrulaması."""

import logging

import pandas as pd

from churn_mlops.config.settings import DataConfig
from churn_mlops.data.exceptions import DataValidationError

logger = logging.getLogger(__name__)


class DataLoader:
    """Ham churn verisini okuyup yapısal olarak doğrular."""

    def __init__(self, config: DataConfig) -> None:
        self._config = config

    def load(self) -> pd.DataFrame:
        """Ham veriyi okur, doğrular ve DataFrame olarak döndürür."""
        path = self._config.raw_path
        if not path.exists():
            raise FileNotFoundError(
                f"Raw data not found: {path}. Did you run 'dvc pull'?"
            )

        logger.info("Loading raw data from %s", path)
        df = pd.read_csv(path)
        self._validate(df)
        logger.info("Loaded %d rows and %d columns", df.shape[0], df.shape[1])
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        """Verinin yapısal kurallara uyduğunu kontrol eder."""
        if df.empty:
            raise DataValidationError("Raw data is empty.")

        required_columns = {self._config.id_column, self._config.target_column}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise DataValidationError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        duplicate_count = df[self._config.id_column].duplicated().sum()
        if duplicate_count > 0:
            raise DataValidationError(
                f"Found {duplicate_count} duplicate values "
                f"in '{self._config.id_column}'."
            )
