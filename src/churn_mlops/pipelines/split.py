"""DVC 'split' aşaması: temizlenmiş veriyi eğitim ve test kümelerine ayırır."""

import logging

import pandas as pd

from churn_mlops.config import ProjectConfig
from churn_mlops.data import DataSplitter
from churn_mlops.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def run(config: ProjectConfig) -> None:
    """Ayırma aşamasını verilen konfigürasyonla çalıştırır."""
    clean_df = pd.read_csv(config.data.interim_path)
    train_df, test_df = DataSplitter(config.data, config.split).split(clean_df)

    outputs = {
        config.data.train_path: train_df,
        config.data.test_path: test_df,
    }
    for path, df in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved %d rows to %s", len(df), path)


def main() -> None:
    """Komut satırı giriş noktası."""
    setup_logging()
    run(ProjectConfig.from_yaml())


if __name__ == "__main__":
    main()
