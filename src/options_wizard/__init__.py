from .parallel import *
from .model import *
from .data import *
from .backtest import *
from .universe import *
from .position import *

import types
import logging
from pathlib import Path

class DefaultTickNameFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "tick_name"):
            record.tick_name = "-"   # default
        return True

def caller_stem(stacklevel: int = 2) -> str:
    import inspect
    from pathlib import Path

    frame = inspect.currentframe()
    try:
        f = frame
        for _ in range(stacklevel):
            f = f.f_back
            if f is None:
                return "unknown"

        filename = f.f_code.co_filename
        stem = Path(filename).stem

        if stem.isdigit():
            try:
                from ipykernel import connect
                return Path(connect.get_connection_file()).stem
            except Exception:
                return "notebook"

        return stem
    finally:
        del frame


def set_log(level: str, log_file: str | None = None) -> None:
    level = level.upper()
    numeric_level = logging._nameToLevel.get(level)
    if numeric_level is None:
        raise ValueError(f"Invalid log level '{level}'")

    if log_file is None:
        log_file = "logfile.log"
    log_path = Path.cwd() / "tmp" / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("[%(asctime)s | %(levelname)s | %(tick_name)s]: %(message)s")
    filter = DefaultTickNameFilter()

    handlers = [
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ]

    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
    if not any(isinstance(f, DefaultTickNameFilter) for f in root_logger.filters):
        root_logger.addFilter(filter)

    # Apply formatter and filter to all loggers (including libraries)
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        for handler in logger.handlers:
            handler.setFormatter(formatter)
        if not any(isinstance(f, DefaultTickNameFilter) for f in logger.filters):
            logger.addFilter(filter)
            
__all__ = sorted(
    name
    for name, val in globals().items()
    if not name.startswith("_")
    and name not in {"types", "logging", "Path"}
    and not isinstance(val, types.ModuleType)
)

