import dataclasses
import enum
import subprocess
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, override

import hydra
import loguru
import omegaconf
from hydra.core.config_store import ConfigStore
from rich.console import Console
from rich.logging import RichHandler


class TsubameNodeType(enum.StrEnum):
    NODE_F = enum.auto()
    NODE_H = enum.auto()
    NODE_Q = enum.auto()
    NODE_O = enum.auto()
    GPU_1 = enum.auto()
    GPU_H = enum.auto()
    CPU_160 = enum.auto()
    CPU_80 = enum.auto()
    CPU_40 = enum.auto()
    CPU_16 = enum.auto()
    CPU_8 = enum.auto()
    CPU_4 = enum.auto()


@dataclasses.dataclass
class SchedulerConfig(ABC):
    # Parameters for command formatting (used in multirun sweeps)
    command: str = omegaconf.MISSING
    params: dict[str, Any] = dataclasses.field(
        default_factory=dict,
    )

    @abstractmethod
    def make_job_script(self, **params: Any) -> str:
        pass

    @abstractmethod
    def make_submit_command(self) -> list[str]:
        pass


@dataclasses.dataclass
class TsubameConfig(SchedulerConfig):
    # Parameters for batch script
    node_type: TsubameNodeType = omegaconf.MISSING
    num_nodes: int = omegaconf.MISSING
    num_gpus: int = omegaconf.MISSING
    walltime: str = omegaconf.MISSING
    priority: int = omegaconf.MISSING
    job_name: str = omegaconf.MISSING
    log_file: Path = omegaconf.MISSING
    reserve_id: str | None = None
    env_modules: list[str] | None = None
    env_variables: list[str] | None = None

    # Parameters for job submission
    group_name: str = omegaconf.MISSING
    script_file: Path = omegaconf.MISSING
    subscription: bool = False

    @override
    def make_job_script(self, **params: Any) -> str:
        job_script = """
            #!/bin/bash

            #$ -cwd
            #$ -l {node_type}={num_nodes}
            #$ -l h_rt={walltime}
            #$ -p {priority}
            #$ -N {job_name}
            #$ -o {log_file}
            #$ -j y
            {reserve_id}

            # Load environment modules.
            module purge
            {env_modules}

            # Set environment variables.
            {env_variables}

            mpiexec -npernode {num_gpus} -n {num_procs} {command}
        """
        job_script = textwrap.dedent(job_script)
        job_script = job_script.format(
            **dict(
                dataclasses.asdict(self),
                reserve_id=f"#$ -ar {self.reserve_id}" if self.reserve_id else "",
                env_modules=" && ".join(
                    f"module load {env_module}" for env_module in self.env_modules or []
                ),
                env_variables=" && ".join(
                    f"export {env_variable}" for env_variable in self.env_variables or []
                ),
                num_procs=self.num_nodes * self.num_gpus,
                command=self.command.format(**dict(self.params, **params)),
            ),
        )
        return job_script

    @override
    def make_submit_command(self) -> list[str]:
        command = ["qsub", "-g", self.group_name, self.script_file]
        return command


@dataclasses.dataclass
class BaseConfig:
    scheduler: SchedulerConfig = omegaconf.MISSING
    # Parameters for WandB & nerfstudio
    output_dir: Path = omegaconf.MISSING
    method_name: str = omegaconf.MISSING
    project_name: str = omegaconf.MISSING
    experiment_name: str = omegaconf.MISSING
    timestamp: str = omegaconf.MISSING


# Register structured config as schema
config_store = ConfigStore.instance()
config_store.store(name="_base_", node=BaseConfig)
config_store.store(group="scheduler", name="_tsubame_", node=TsubameConfig)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: omegaconf.DictConfig) -> None:
    console = Console()

    loguru.logger.remove()
    loguru.logger.add(
        sink=RichHandler(
            console=console,
            show_level=False,
            show_path=False,
            show_time=False,
            rich_tracebacks=True,
        ),
        colorize=False,
        diagnose=True,
        backtrace=True,
    )

    config: BaseConfig = omegaconf.OmegaConf.to_object(config)
    scheduler: TsubameConfig = config.scheduler

    job_script = scheduler.make_job_script(**dataclasses.asdict(config))

    script_file = scheduler.script_file.expanduser()
    script_file.parent.mkdir(parents=True, exist_ok=True)
    with open(script_file, "w") as fp:
        fp.write(job_script)

    loguru.logger.success(f"📝 Job script created: <{script_file}>")

    command = scheduler.make_submit_command()
    loguru.logger.info(f"⏳ Submitting job: <{' '.join(map(str, command))}>")

    try:
        result = subprocess.run(
            args=command,
            capture_output=True,
            check=True,
            text=True,
        )
        loguru.logger.success("🚀 Job submitted successfully!")
        console.rule()
        console.print(result.stdout.strip())
        console.rule()

    except subprocess.CalledProcessError as error:
        loguru.logger.error("⚠️ Job submission failed!")
        console.rule()
        console.print(error.stderr.strip())
        console.rule()
        raise


if __name__ == "__main__":
    main()
