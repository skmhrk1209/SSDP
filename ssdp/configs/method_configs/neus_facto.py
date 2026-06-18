import dataclasses

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.base_surface_model import SurfaceModelConfig
from nerfstudio.plugins.types import MethodSpecification
from ssdp.fields import SDFConfig
from ssdp.models import NeuSFactoConfig

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
            model=NeuSFactoConfig(
                **dict(
                    _asdict(target_model_config),
                    sdf_field=SDFConfig(
                        **dict(
                            _asdict(
                                target_model_config.sdf_field,
                                excluded_keys=["beta_init"],
                            ),
                            bias=0.2,
                            inside_outside=False,
                        ),
                    ),
                    near_plane=0.0,
                    far_plane=5.0,
                    background_model="mlp",
                    nlml_color_loss_mult=0.0,
                    volume_color_loss_mult=1.0,
                    anneal_end_step=source_trainer_config.max_num_iterations // 2,
                ),
            ),
        ),
    )
    return trainer_config


config = MethodSpecification(
    config=_create_trainer_config(
        method_name="neus-facto",
        source_trainer_config=method_configs["neus-facto"],
    ),
    description="NeuS-Facto",
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
            method_name=f"neus-facto-{suffix}",
            source_trainer_config=method_configs["neus-facto"],
            scale_factor=scale_factor,
        ),
        description=f"NeuS-Facto with x{scale_factor} iterations",
    )
