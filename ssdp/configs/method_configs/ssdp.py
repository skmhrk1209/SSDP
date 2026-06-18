import dataclasses

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.base_surface_model import SurfaceModelConfig
from nerfstudio.plugins.types import MethodSpecification
from ssdp.fields import SSDPConfig
from ssdp.models import NeuSConfig

from ._base import _create_trainer_config as _create_base_trainer_config
from ._utils import _asdict


def _create_trainer_config(
    method_name: str,
    source_trainer_config: TrainerConfig,
    target_trainer_config: TrainerConfig | None = None,
    scale_factor: float = 1.0,
) -> TrainerConfig:
    source_trainer_config = _create_base_trainer_config(
        base_trainer_config=source_trainer_config,
        scale_factor=scale_factor,
    )
    target_trainer_config = target_trainer_config or source_trainer_config
    target_model_config: SurfaceModelConfig = target_trainer_config.pipeline.model
    trainer_config = dataclasses.replace(
        source_trainer_config,
        method_name=method_name,
        pipeline=dataclasses.replace(
            source_trainer_config.pipeline,
            model=NeuSConfig(
                **dict(
                    _asdict(target_model_config),
                    sdf_field=SSDPConfig(
                        **dict(
                            _asdict(target_model_config.sdf_field),
                            num_layers_ou=target_model_config.sdf_field.num_layers_color,
                            bias=0.2,
                            inside_outside=False,
                        ),
                    ),
                    near_plane=0.0,
                    far_plane=5.0,
                    background_model="mlp",
                    nlml_color_loss_mult=1.0,
                    volume_color_loss_mult=0.0,
                ),
            ),
        ),
    )
    return trainer_config


config = MethodSpecification(
    config=_create_trainer_config(
        method_name="ssdp",
        source_trainer_config=method_configs["neus-facto"],
        target_trainer_config=method_configs["neus"],
    ),
    description="SSDP",
)

for suffix, scale_factor in dict(
    base=1.0,
    half=0.5,
    double=2.0,
    quarter=0.25,
    quadruple=4.0,
).items():
    globals()[f"config_{suffix}"] = MethodSpecification(
        config=_create_trainer_config(
            method_name=f"ssdp-{suffix}",
            source_trainer_config=method_configs["neus-facto"],
            target_trainer_config=method_configs["neus"],
            scale_factor=scale_factor,
        ),
        description=f"SSDP with x{scale_factor} iterations",
    )
