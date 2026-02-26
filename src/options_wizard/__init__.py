from .parallel import *
from .model import *
from .data import *
from .backtest import *
from .universe import *
from .position import *

import types
import logging
from pathlib import Path
import structlog
from structlog.stdlib import ProcessorFormatter

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

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
    ]

    console_formatter = ProcessorFormatter(
        processors=[
            ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        foreign_pre_chain=shared_processors,
    )
    file_formatter = ProcessorFormatter(
        processors=[
            ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    handlers: list[logging.Handler] = [file_handler, console_handler]

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        try:
            handler.close()
        except Exception:
            pass
    root_logger.handlers = []
    root_logger.setLevel(numeric_level)
    for handler in handlers:
        root_logger.addHandler(handler)

    # Allow child loggers to propagate to root handlers.
    for name in logging.root.manager.loggerDict:
        existing = logging.root.manager.loggerDict[name]
        if isinstance(existing, logging.Logger):
            for handler in existing.handlers:
                try:
                    handler.close()
                except Exception:
                    pass
            existing.handlers = []
            existing.propagate = True

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
            
__all__ = sorted(
    name
    for name, val in globals().items()
    if not name.startswith("_")
    and name
    not in {
        "types",
        "logging",
        "Path",
        "structlog",
        "ProcessorFormatter",
    }
    and not isinstance(val, types.ModuleType)
)
