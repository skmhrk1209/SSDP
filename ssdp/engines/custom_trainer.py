import dataclasses
import json
from pathlib import Path
from typing import Any, override

import torch
import tyro

from nerfstudio.engine.trainer import Trainer, TrainerConfig
from ssdp.utils import distributed, git


@dataclasses.dataclass
class GitConfig:
    branch: str = dataclasses.field(default_factory=git.get_branch)
    commit_id: str = dataclasses.field(default_factory=git.get_commit_id)
    remote_url: str = dataclasses.field(default_factory=git.get_remote_url)


@dataclasses.dataclass
class BackendConfig:
    benchmark: bool = False
    deterministic: bool = True


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
    git: tyro.conf.Fixed[GitConfig] = dataclasses.field(
        default_factory=GitConfig,
    )
    backend: BackendConfig = dataclasses.field(
        default_factory=BackendConfig,
    )
    distributed: DistributedConfig = dataclasses.field(
        default_factory=DistributedConfig,
    )
    output_subdir: str = "nerfstudio"
    anomaly_detection: bool = False

    @override
    def get_base_dir(self) -> Path:
        super().get_base_dir()
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
