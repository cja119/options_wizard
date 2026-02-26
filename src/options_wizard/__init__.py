from .parallel import *
from .model import *
from .data import *
from .backtest import *
from .universe import *
from .position import *

import types
import logging
import structlog

_LOGGING_CONFIGURED = False

def configure_logging(level: str = "INFO", force: bool = False) -> None:
    """Configure structlog for the package."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED and not force:
        return

    level = level.upper()
    numeric_level = logging._nameToLevel.get(level)
    if numeric_level is None:
        raise ValueError(f"Invalid log level '{level}'")

    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        force=force,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    _LOGGING_CONFIGURED = True


def set_log(level: str, log_file: str | None = None) -> None:
    """Backward-compatible alias for older code paths."""
    _ = log_file
    configure_logging(level, force=True)


# Default setup so importing the package is enough for normal usage.
configure_logging()


__all__ = sorted(
    name
    for name, val in globals().items()
    if not name.startswith("_")
    and name
    not in {
        "types",
        "logging",
        "structlog",
    }
    and not isinstance(val, types.ModuleType)
)
