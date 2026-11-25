import dataclasses
from pathlib import Path

from nerfstudio.engine.trainer import TrainerConfig
from ssdp.engines import CustomTrainerConfig

from ._utils import asdict


def create_trainer_config(base_trainer_config: TrainerConfig) -> CustomTrainerConfig:
    trainer_config = CustomTrainerConfig(
        **dict(
            asdict(base_trainer_config),
            method_name=CustomTrainerConfig.method_name,
            project_name=CustomTrainerConfig.project_name,
            experiment_name=CustomTrainerConfig.experiment_name,
            steps_per_save=1000,
            steps_per_eval_batch=1000,
            steps_per_eval_image=1000,
            logging=dataclasses.replace(
                base_trainer_config.logging,
                relative_log_dir=Path("../"),
            ),
            relative_model_dir="checkpoints",
            vis="wandb",
        ),
    )
    return trainer_config
