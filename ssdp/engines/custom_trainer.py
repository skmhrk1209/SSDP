import dataclasses
import json
from pathlib import Path
from typing import Any, override

import torch
from tyro.conf import Fixed

from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
    SchedulerConfig,
)
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from ssdp.utils import distributed, git
from ssdp.utils.jaxtyping import is_jaxtyping_enabled
from ssdp.utils.types import OptimizerGroupConfig


@dataclasses.dataclass(frozen=True)
class GitConfig:
    branch: str = dataclasses.field(default_factory=git.get_branch)
    commit_id: str = dataclasses.field(default_factory=git.get_commit_id)
    remote_url: str = dataclasses.field(default_factory=git.get_remote_url)


@dataclasses.dataclass
class BackendConfig:
    benchmark: bool = False
    deterministic: bool = True
    cudnn_allow_tf32: bool = True
    cublas_allow_tf32: bool = False
    cublas_workspace_config: str = ":4096:8"


@dataclasses.dataclass
class DistributedConfig:
    backend: distributed.Backend = distributed.Backend.NCCL
    init_method: distributed.InitMethod = distributed.InitMethod.ENV
    master_addr: str = dataclasses.field(default_factory=distributed.get_master_address)
    master_port: int = dataclasses.field(default_factory=distributed.get_master_port)
    global_size: int = dataclasses.field(default_factory=distributed.get_global_size)
    global_rank: int = dataclasses.field(default_factory=distributed.get_global_rank)
    local_size: int = dataclasses.field(default_factory=distributed.get_local_size)
    local_rank: int = dataclasses.field(default_factory=distributed.get_local_rank)


@dataclasses.dataclass
class CustomTrainerConfig(TrainerConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: CustomTrainer,
    )
    git: Fixed[GitConfig] = dataclasses.field(
        default_factory=GitConfig,
    )
    jaxtyping: Fixed[bool] = dataclasses.field(
        default_factory=is_jaxtyping_enabled,
    )
    backend: BackendConfig = dataclasses.field(
        default_factory=BackendConfig,
    )
    distributed: DistributedConfig = dataclasses.field(
        default_factory=DistributedConfig,
    )
    schedule_scale_factor: float = dataclasses.field(
        default=1.0,
        repr=False,
    )
    scale_warmup_schedule: bool = dataclasses.field(
        default=False,
        repr=False,
    )
    scale_steps_per_save: bool = dataclasses.field(
        default=False,
        repr=False,
    )
    scale_steps_per_eval: bool = dataclasses.field(
        default=False,
        repr=False,
    )
    method_name: str = "neus"
    project_name: str = "project"
    experiment_name: str = "experiment"
    output_subdir: str = "nerfstudio"
    anomaly_detection: bool = False

    def __post_init__(self) -> None:
        self.max_num_iterations = self._scale_num_steps(self.max_num_iterations)
        if self.scale_steps_per_save:
            self.steps_per_save = self._scale_num_steps(self.steps_per_save)
        if self.scale_steps_per_eval:
            self.steps_per_eval_batch = self._scale_num_steps(self.steps_per_eval_batch)
            self.steps_per_eval_image = self._scale_num_steps(self.steps_per_eval_image)
            self.steps_per_eval_all_images = self._scale_num_steps(self.steps_per_eval_all_images)
        self.optimizers = {
            name: OptimizerGroupConfig(
                optimizer=optimizer_group_config["optimizer"],
                scheduler=self._scale_schedule(optimizer_group_config["scheduler"]),
            )
            for name, optimizer_group_config in self.optimizers.items()
        }
        self.schedule_scale_factor = 1.0
        self.scale_warmup_schedule = False
        self.scale_steps_per_save = False
        self.scale_steps_per_eval = False

    def _scale_num_steps(self, num_steps: int) -> int:
        return round(num_steps * self.schedule_scale_factor)

    def _scale_schedule(self, scheduler_config: SchedulerConfig) -> SchedulerConfig:
        if isinstance(
            scheduler_config,
            MultiStepSchedulerConfig
            | ExponentialDecaySchedulerConfig
            | CosineDecaySchedulerConfig,
        ):
            scheduler_config = dataclasses.replace(
                scheduler_config,
                max_steps=round(scheduler_config.max_steps * self.schedule_scale_factor),
            )
            if isinstance(scheduler_config, MultiStepSchedulerConfig):
                scheduler_config = dataclasses.replace(
                    scheduler_config,
                    milestones=tuple(
                        round(milestone * self.schedule_scale_factor)
                        for milestone in scheduler_config.milestones
                    ),
                )
            if self.scale_warmup_schedule:
                if isinstance(scheduler_config, ExponentialDecaySchedulerConfig):
                    scheduler_config = dataclasses.replace(
                        scheduler_config,
                        warmup_steps=round(
                            scheduler_config.warmup_steps * self.schedule_scale_factor
                        ),
                    )
                if isinstance(scheduler_config, CosineDecaySchedulerConfig):
                    scheduler_config = dataclasses.replace(
                        scheduler_config,
                        warm_up_end=round(
                            scheduler_config.warm_up_end * self.schedule_scale_factor
                        ),
                    )
        else:
            raise TypeError(f"Unsupported `SchedulerConfig`: {scheduler_config}")
        return scheduler_config

    @override
    def get_base_dir(self) -> Path:
        base_dir = (
            self.output_dir
            / self.project_name
            / self.method_name
            / self.experiment_name
            / self.timestamp
            / self.output_subdir
        )
        return base_dir


class CustomTrainer(Trainer):
    config: CustomTrainerConfig

    @override
    def train_iteration(
        self,
        step: int,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        with torch.autograd.set_detect_anomaly(self.config.anomaly_detection):
            loss, losses, metrics = super().train_iteration(step)
        if not torch.isfinite(loss):
            info = dict(zip(losses.keys(), map(torch.Tensor.item, losses.values()), strict=True))
            info = json.dumps(info, indent=4, sort_keys=True)
            raise RuntimeError(f"Non-finite loss detected at step {step}: {info}")
        return loss, losses, metrics

    @torch.no_grad()
    @override
    def eval_iteration(self, *args: Any, **kwargs: Any) -> Any:
        return super().eval_iteration(*args, **kwargs)
