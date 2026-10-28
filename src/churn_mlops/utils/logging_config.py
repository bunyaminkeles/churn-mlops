"""Uygulama genelinde logging yapılandırması."""

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Kök logger'ı tutarlı bir formatla yapılandırır."""
    logging.basicConfig(level=level, format=LOG_FORMAT)
