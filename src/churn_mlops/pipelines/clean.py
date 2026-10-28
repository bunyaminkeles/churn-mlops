"""DVC 'clean' aşaması: ham veriyi yükler, temizler ve ara katmana yazar."""

import logging

from churn_mlops.config import ProjectConfig
from churn_mlops.data import DataCleaner, DataLoader
from churn_mlops.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def run(config: ProjectConfig) -> None:
    """Temizleme aşamasını verilen konfigürasyonla çalıştırır."""
    raw_df = DataLoader(config.data).load()
    clean_df = DataCleaner(config.data).clean(raw_df)

    output_path = config.data.interim_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    logger.info("Saved cleaned data to %s", output_path)


def main() -> None:
    """Komut satırı giriş noktası."""
    setup_logging()
    run(ProjectConfig.from_yaml())


if __name__ == "__main__":
    main()
