from collections.abc import Callable
from typing import Any


def signature_of[**P, R](
    function: Callable[P, R],
) -> Callable[[Callable[..., Any]], Callable[P, R]]:
    def decorator(function: Callable[..., Any]) -> Callable[P, R]:
        return function

    return decorator
