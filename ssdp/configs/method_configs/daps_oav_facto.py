import dataclasses

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.base_surface_model import SurfaceModelConfig
from nerfstudio.plugins.types import MethodSpecification
from ssdp.fields import LDMOaVConfig
from ssdp.models import DAPSNeuSFactoConfig
from ssdp.optimizers import PSGHMCConfig

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
    model_config = DAPSNeuSFactoConfig(
        **dict(
            _asdict(target_model_config),
            sdf_field=LDMOaVConfig(
                **dict(
                    _asdict(
                        target_model_config.sdf_field,
                        excluded_keys=["beta_init"],
                    ),
                    bias=0.5,
                    inside_outside=False,
                ),
            ),
            near_plane=0.0,
            far_plane=5.0,
            background_model="mlp",
            eikonal_loss_mult=0.0,
            nll_color_loss_mult=1.0,
            nlml_color_loss_mult=0.0,
        ),
    )
    trainer_config = dataclasses.replace(
        source_trainer_config,
        method_name=method_name,
        pipeline=dataclasses.replace(
            source_trainer_config.pipeline,
            model=model_config,
        ),
        optimizers=dict(
            source_trainer_config.optimizers,
            field=dict(
                optimizer=PSGHMCConfig(lr=1.0e-4),
                scheduler=None,
            ),
            latent=dict(
                optimizer=PSGHMCConfig(lr=1.0e-4),
                scheduler=None,
            ),
            nuisance=source_trainer_config.optimizers.pop("fields"),
        ),
    )
    return trainer_config


for suffix, scale_factor in dict(
    base=1.0,
    half=0.5,
    double=2.0,
    quarter=0.25,
    quadruple=4.0,
).items():
    globals()[f"config_{suffix}"] = MethodSpecification(
        config=_create_trainer_config(
            method_name=f"daps-oav-facto-{suffix}",
            source_trainer_config=method_configs["neus-facto"],
            scale_factor=scale_factor,
        ),
        description=f"Latent-DAPS on OaV-Facto with x{scale_factor} iterations",
    )
