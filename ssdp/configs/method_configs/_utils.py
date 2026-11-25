import dataclasses
from typing import Any


def asdict(dataclass: Any) -> dict[str, Any]:
    dictionary = {
        field.name: getattr(dataclass, field.name)
        for field in dataclasses.fields(dataclass)
        if field.name != "_target"
    }
    return dictionary
