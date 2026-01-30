from __future__ import annotations

from pathlib import Path
from threading import Lock


_LOG_PATH = Path("tmp") / "backtest_errors.log"
_WARNING_COUNT = 0
_BUFFER: list[str] = []
_BUFFER_LIMIT = 200
_LOCK = Lock()


def reset(clear_log: bool = True) -> None:
    global _WARNING_COUNT
    _WARNING_COUNT = 0
    _BUFFER.clear()
    if clear_log:
        try:
            _LOG_PATH.write_text("", encoding="utf-8")
        except Exception:
            # If we can't clear the log, keep going and append.
            pass


def _flush_locked() -> None:
    if not _BUFFER:
        return
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(_BUFFER) + "\n")
    except Exception:
        # Logging should never stop the backtest.
        pass
    _BUFFER.clear()


def log_same_day_exit(message: str) -> int:
    global _WARNING_COUNT
    with _LOCK:
        _WARNING_COUNT += 1
        _BUFFER.append(message.rstrip())
        if len(_BUFFER) >= _BUFFER_LIMIT:
            _flush_locked()
        return _WARNING_COUNT


def warning_count() -> int:
    return _WARNING_COUNT


def log_path() -> Path:
    return _LOG_PATH


def finalize() -> int:
    with _LOCK:
        _flush_locked()
        count = _WARNING_COUNT
    print(f"{count} warnings loaded into the log file")
    return count
