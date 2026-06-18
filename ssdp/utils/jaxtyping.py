from collections.abc import Callable
from typing import Any

import jaxtyping as jt
import loguru
from beartype import beartype


def _is_inner_function(function: Callable[..., Any]) -> bool:
    return "<locals>" in function.__qualname__


def jaxtyped[C: Callable[..., Any], CT: Callable[..., Any] | type](
    typechecker: Callable[[C], C] = beartype,
    disable_inner_typecheck: bool = True,
    **kwargs: Any,
) -> Callable[[CT], CT]:
    def _decorator(
        callable_or_type: CT,
        typechecker: Callable[[C], C] = typechecker,
    ) -> CT:
        if _is_inner_function(callable_or_type) and disable_inner_typecheck:
            loguru.logger.bind(once=True).warning(
                f"Runtime type checking has been disabled for the inner function `{callable_or_type.__qualname__}`, "
                f"since applying `jaxtyping.jaxtyped` to an inner function makes each warning repeated at each iteration. "
                f"Please specify `disable_inner_typecheck=False` if this behavior is undesirable."
            )
            typechecker = None

        return jt.jaxtyped(
            fn=callable_or_type,
            typechecker=typechecker,
            **kwargs,
        )

    return _decorator
