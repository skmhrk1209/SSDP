import dataclasses

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.base_surface_model import SurfaceModelConfig
from nerfstudio.plugins.types import MethodSpecification
from ssdp.fields import UNISConfig, UNISKernelType
from ssdp.models import NeuSFactoConfig

from ._base import _create_trainer_config as _create_base_trainer_config
from ._utils import _asdict


def _create_trainer_config(
    method_name: str,
    source_trainer_config: TrainerConfig,
    target_trainer_config: TrainerConfig | None = None,
    scale_factor: float = 1.0,
    kernel_type: UNISKernelType = UNISKernelType.LAPLACE,
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
                    sdf_field=UNISConfig(
                        **dict(
                            _asdict(
                                target_model_config.sdf_field,
                                excluded_keys=["beta_init"],
                            ),
                            bias=0.2,
                            inside_outside=False,
                            kernel_type=kernel_type,
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


# UNIS case studies with k = 0 and an SDF field, i.e., UNIS-C1/C2/C3 (Eqs. 20-22).
for case_study, kernel_type in dict(
    c1=UNISKernelType.LAPLACE,
    c2=UNISKernelType.ALGEBRAIC,
    c3=UNISKernelType.SOFTPLUS,
).items():
    globals()[f"config_{case_study}"] = MethodSpecification(
        config=_create_trainer_config(
            method_name=f"unis-facto-{case_study}",
            source_trainer_config=method_configs["neus-facto"],
            kernel_type=kernel_type,
        ),
        description=f"UNIS-Facto ({case_study.upper()}: {kernel_type} kernel)",
    )

    for suffix, scale_factor in dict(
        base=1.0,
        half=0.5,
        double=2.0,
        quarter=0.25,
        quadruple=4.0,
    ).items():
        globals()[f"config_{case_study}_{suffix}"] = MethodSpecification(
            config=_create_trainer_config(
                method_name=f"unis-facto-{case_study}-{suffix}",
                source_trainer_config=method_configs["neus-facto"],
                scale_factor=scale_factor,
                kernel_type=kernel_type,
            ),
            description=f"UNIS-Facto ({case_study.upper()}: {kernel_type} kernel) with x{scale_factor} iterations",
        )
