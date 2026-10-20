import dataclasses

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.engine.schedulers import MultiStepSchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.base_surface_model import SurfaceModelConfig
from nerfstudio.plugins.types import MethodSpecification
from ssdp.fields import LDMSSDPConfig
from ssdp.models import DAPSNeuSFactoConfig
from ssdp.optimizers import PSGLDConfig

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
            sdf_field=LDMSSDPConfig(
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
            eikonal_loss_mult=0.0,
            nll_color_loss_mult=0.0,
            nlml_color_loss_mult=1.0,
        ),
    )
    num_steps_train = source_trainer_config.max_num_iterations
    num_steps_mcmc = num_steps_train // model_config.num_steps_anneal
    trainer_config = dataclasses.replace(
        source_trainer_config,
        method_name=method_name,
        pipeline=dataclasses.replace(
            source_trainer_config.pipeline,
            model=model_config,
        ),
        optimizers=dict(
            source_trainer_config.optimizers,
            mcmc=dict(
                optimizer=PSGLDConfig(lr=1.0e-4),
                scheduler=MultiStepSchedulerConfig(
                    max_steps=num_steps_train,
                    gamma=0.01 ** (1.0 / (model_config.num_steps_anneal - 1)),
                    milestones=tuple(range(num_steps_mcmc, num_steps_train, num_steps_mcmc)),
                ),
            ),
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
            method_name=f"daps-ssdp-facto-{suffix}",
            source_trainer_config=method_configs["neus-facto"],
            scale_factor=scale_factor,
        ),
        description=f"Latent-DAPS on SSDP-Facto with x{scale_factor} iterations",
    )
