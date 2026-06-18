"""Local, file-based persistence for traces.

v1 is local-only by design (see the project brief): no server, no cloud DB.
JSON is the default store because the Phase 0 gate is "open it and read it" —
a human-legible file beats an opaque row. A SQLite store can slot in behind the
same `TraceStore` shape in Phase 3 without touching callers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from .schema import Trace

DEFAULT_DIR = Path(os.environ.get("AUTOPSY_DIR", ".autopsy")) / "traces"


@runtime_checkable
class TraceStore(Protocol):
    """The persistence contract. Swap implementations without touching callers."""

    def save(self, trace: Trace) -> str: ...
    def load(self, trace_id: str) -> Trace: ...
    def list(self) -> list[dict[str, Any]]: ...


class JSONStorage:
    """One ``<trace_id>.json`` file per trace under a directory."""

    def __init__(self, directory: str | os.PathLike[str] = DEFAULT_DIR) -> None:
        self.directory = Path(directory)

    def _path(self, trace_id: str) -> Path:
        return self.directory / f"{trace_id}.json"

    def save(self, trace: Trace) -> str:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(trace.trace_id)
        # Write to a temp file then atomically replace, so a reader never sees
        # a half-written trace.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return str(path)

    def load(self, trace_id: str) -> Trace:
        path = self._path(trace_id)
        if not path.exists():
            raise FileNotFoundError(f"No trace found for id {trace_id!r} in {self.directory}")
        return Trace.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        """Return trace summaries, newest first. Skips unreadable files."""
        if not self.directory.exists():
            return []
        summaries: list[dict[str, Any]] = []
        for path in self.directory.glob("*.json"):
            try:
                trace = Trace.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            summaries.append(trace.summary())
        summaries.sort(key=lambda s: s["start_time"], reverse=True)
        return summaries


# --- module-level default store + convenience functions ---------------------

_default_store: TraceStore = JSONStorage()


def get_default_store() -> TraceStore:
    return _default_store


def set_default_store(store: TraceStore) -> None:
    """Point Autopsy's convenience functions and `start_run` at a new store."""
    global _default_store
    _default_store = store


def save_trace(trace: Trace, store: Optional[TraceStore] = None) -> str:
    return (store or _default_store).save(trace)


def load_trace(trace_id: str, store: Optional[TraceStore] = None) -> Trace:
    return (store or _default_store).load(trace_id)


def list_traces(store: Optional[TraceStore] = None) -> list[dict[str, Any]]:
    return (store or _default_store).list()
