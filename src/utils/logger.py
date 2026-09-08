"""Structured logging utilities."""

import datetime
import json
import os
import sys


class SatQueryLogger:
    """Lightweight structured logger with JSON output support."""

    def __init__(self, name: str = "satquery", log_file: str = None, level: str = "INFO"):
        self.name = name
        self.log_file = log_file
        self.level = level
        self._levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        if log_file:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    def _log(self, level: str, message: str, **kwargs):
        if self._levels.get(level, 0) < self._levels.get(self.level, 20):
            return
        ts = datetime.datetime.utcnow().isoformat() + "Z"
        entry = {"timestamp": ts, "level": level, "name": self.name, "message": message}
        entry.update(kwargs)
        line = json.dumps(entry)
        print(line, file=sys.stderr if level in ("WARNING", "ERROR") else sys.stdout)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(line + "\n")

    def debug(self, msg, **kw):   self._log("DEBUG",   msg, **kw)
    def info(self, msg, **kw):    self._log("INFO",    msg, **kw)
    def warning(self, msg, **kw): self._log("WARNING", msg, **kw)
    def error(self, msg, **kw):   self._log("ERROR",   msg, **kw)


def get_logger(name: str = "satquery", log_file: str = None) -> SatQueryLogger:
    return SatQueryLogger(name=name, log_file=log_file)
