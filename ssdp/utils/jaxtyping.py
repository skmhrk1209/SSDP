import os
from collections.abc import Callable
from typing import Any, cast

import jaxtyping as jt
from beartype import beartype

_JAXTYPING_DISABLE_KEY = "JAXTYPING_DISABLE"


def is_jaxtyping_enabled() -> bool:
    return not bool(int(os.environ.get(_JAXTYPING_DISABLE_KEY, "1")))


def _is_inner_function(function: Callable[..., Any]) -> bool:
    return "<locals>" in function.__qualname__


def jaxtyped[C: Callable[..., Any], CT: Callable[..., Any] | type](
    typechecker: Callable[[C], C] | None = beartype,
    enabled: bool | None = None,
) -> Callable[[CT], CT]:
    def decorator(
        callable_or_type: CT,
        typechecker: Callable[[C], C] | None = typechecker,
        enabled: bool | None = enabled,
    ) -> CT:
        if enabled is None:
            enabled = is_jaxtyping_enabled() and not _is_inner_function(callable_or_type)

        if not enabled:
            return callable_or_type

        return cast(
            CT,
            jt.jaxtyped(
                fn=callable_or_type,
                typechecker=typechecker,
            ),
        )

    return decorator
