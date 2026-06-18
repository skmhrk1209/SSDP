import dataclasses
from pathlib import Path

from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
    SchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from ssdp.engines import CustomTrainerConfig

from ._utils import _asdict


def _create_scheduler_config(
    base_scheduler_config: SchedulerConfig,
    scale_factor: float = 1.0,
) -> SchedulerConfig:
    if isinstance(
        base_scheduler_config,
        MultiStepSchedulerConfig | ExponentialDecaySchedulerConfig | CosineDecaySchedulerConfig,
    ):
        scheduler_config = dataclasses.replace(
            base_scheduler_config,
            max_steps=round(base_scheduler_config.max_steps * scale_factor),
        )
        if isinstance(scheduler_config, MultiStepSchedulerConfig):
            scheduler_config = dataclasses.replace(
                scheduler_config,
                milestones=tuple(
                    round(milestone * scale_factor)
                    for milestone in base_scheduler_config.milestones
                ),
            )
    else:
        raise ValueError(f"Unsupported `SchedulerConfig`: {type(base_scheduler_config)}")
    return scheduler_config


def _create_trainer_config(
    base_trainer_config: TrainerConfig,
    scale_factor: float = 1.0,
) -> TrainerConfig:
    num_iterations = round(base_trainer_config.max_num_iterations * scale_factor)
    trainer_config = CustomTrainerConfig(
        **dict(
            _asdict(base_trainer_config),
            max_num_iterations=num_iterations,
            steps_per_save=num_iterations // 10,
            steps_per_eval_batch=num_iterations // 10,
            steps_per_eval_image=num_iterations // 10,
            optimizers={
                name: dict(
                    optimizer_config,
                    scheduler=_create_scheduler_config(
                        base_scheduler_config=optimizer_config["scheduler"],
                        scale_factor=scale_factor,
                    ),
                )
                for name, optimizer_config in base_trainer_config.optimizers.items()
            },
            logging=dataclasses.replace(
                base_trainer_config.logging,
                relative_log_dir=Path("../"),
            ),
            relative_model_dir="checkpoints",
            vis="wandb",
        ),
    )
    return trainer_config
