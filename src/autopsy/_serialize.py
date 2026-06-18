"""Best-effort conversion of arbitrary Python values into JSON-safe structures.

Instrumentation captures whatever a user's function takes and returns. Those
values can be anything — pydantic models, SDK response objects, dataclasses,
deeply nested dicts. We never want capturing a trace to raise, so this coerces
to something serializable and falls back to ``repr`` rather than failing.
"""

from __future__ import annotations

from typing import Any

try:  # pydantic is a hard dependency, but guard anyway for import order safety
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    BaseModel = ()  # type: ignore[assignment]

_MAX_DEPTH = 8
_MAX_STR = 20_000


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Coerce ``obj`` into a JSON-serializable value, never raising."""
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj if len(obj) <= _MAX_STR else obj[:_MAX_STR] + "…[truncated]"

    if _depth >= _MAX_DEPTH:
        return _stringify(obj)

    if BaseModel and isinstance(obj, BaseModel):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            return _stringify(obj)

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v, _depth + 1) for v in obj]

    # SDK response objects often expose model_dump() or dict()
    for attr in ("model_dump", "to_dict", "dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                dumped = method()
            except Exception:
                continue
            # model_dump may itself return non-JSON values; recurse once
            return to_jsonable(dumped, _depth + 1)

    return _stringify(obj)


def _stringify(obj: Any) -> str:
    try:
        text = repr(obj)
    except Exception:
        text = f"<unrepresentable {type(obj).__name__}>"
    return text if len(text) <= _MAX_STR else text[:_MAX_STR] + "…[truncated]"
