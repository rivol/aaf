import logging
from datetime import datetime

import rich
from rich.markup import escape

CRITICAL = logging.CRITICAL
FATAL = logging.FATAL
ERROR = logging.ERROR
WARNING = logging.WARNING
WARN = logging.WARN
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET

min_level: int = INFO


def set_min_level(level: int) -> None:
    global min_level
    min_level = level
    logging.basicConfig(level=level)


def set_level_from_flags(*, quiet: bool = False, debug: bool = False):
    if debug:
        set_min_level(DEBUG)
    elif quiet:
        set_min_level(WARNING)


class Logger:
    def log(self, level: int, event: str, **kwargs) -> None:
        if level < min_level:
            return

        timestamp = datetime.now().isoformat(" ", timespec="milliseconds")
        level_name = logging.getLevelName(level)
        attributes = " ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
        console = rich.get_console()
        console.print(escape(f"{timestamp}: [{level_name}] {event}  {attributes}"), style="dim", highlight=False)

        if level >= CRITICAL:
            console.print_exception(show_locals=True)

    def debug(self, event: str, **kwargs) -> None:
        self.log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs) -> None:
        self.log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs) -> None:
        self.log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs) -> None:
        self.log(logging.ERROR, event, **kwargs)

    def critical(self, event: str, **kwargs) -> None:
        self.log(logging.CRITICAL, event, **kwargs)


log = Logger()
