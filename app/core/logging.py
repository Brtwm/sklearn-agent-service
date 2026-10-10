import logging
import sys
import structlog


def setup_logging(level: str = "INFO") -> None:
    """Настраивает корневой логгер на stdout (стандартный вывод)"""
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def configure_agent_logging() -> None:
    """Write structured agent events as JSON to stdout."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    )