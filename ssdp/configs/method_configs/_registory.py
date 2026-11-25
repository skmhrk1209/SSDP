from collections.abc import Callable

from nerfstudio.fields.base_field import FieldConfig
from nerfstudio.models.base_model import ModelConfig
from ssdp.utils.types import OptimizersConfig


class _Registry[T]:
    def __init__(self, name: str) -> None:
        self.name = name
        self.registory: dict[str, Callable[[T], T]] = {}

    def register(self, key: str | None = None) -> Callable[[Callable[[T], T]], Callable[[T], T]]:
        def wrapper(factory: Callable[[T], T], key: str | None = key) -> Callable[[T], T]:
            key = key or factory.__module__.rsplit(".", 1)[-1]
            if key in self.registory:
                raise KeyError(f"Key <{key}> already exists in the registory <{self.name}>.")
            self.registory[key] = factory
            return factory

        return wrapper

    def __getitem__(self, key: str) -> Callable[[T], T]:
        return self.registory[key]

    def __contains__(self, key: str) -> bool:
        return key in self.registory


MODEL_REGISTORY: _Registry[ModelConfig] = _Registry("model_registory")
FIELD_REGISTORY: _Registry[FieldConfig] = _Registry("field_registory")
OPTIMIZERS_REGISTORY: _Registry[OptimizersConfig] = _Registry("optimizers_registory")
