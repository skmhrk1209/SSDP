import dataclasses
from typing import Any


def _asdict(
    dataclass: Any,
    excluded_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    dictionary = {
        field.name: getattr(dataclass, field.name)
        for field in dataclasses.fields(dataclass)
        if field.name not in ("_target", *(excluded_keys or ()))
    }
    return dictionary
