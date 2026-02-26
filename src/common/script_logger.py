from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys


def default_log_path(root_dir: Path, job_name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return root_dir / "artifacts" / "logs" / f"{job_name}_{stamp}.log"


class ScriptLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def info(self, message: str) -> None:
        print(message)
        self._fh.write(message + "\n")
        self._fh.flush()

    def warn(self, message: str) -> None:
        print(message, file=sys.stderr)
        self._fh.write(message + "\n")
        self._fh.flush()

    def emit_text(self, text: str, *, is_err: bool = False) -> None:
        if not text:
            return
        stream = sys.stderr if is_err else sys.stdout
        stream.write(text)
        if not text.endswith("\n"):
            stream.write("\n")
        stream.flush()
        self._fh.write(text)
        if not text.endswith("\n"):
            self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
