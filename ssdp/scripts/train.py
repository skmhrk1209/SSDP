import random

import loguru
import numpy as np
import torch
import tyro
import yaml
from rich.logging import RichHandler

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.utils.rich_utils import CONSOLE
from ssdp.engines import CustomTrainer, CustomTrainerConfig
from ssdp.utils import distributed, logging


def main(config: CustomTrainerConfig) -> None:
    # ================================================================
    # Setup the logger.

    loguru.logger.remove()
    loguru.logger.add(
        sink=RichHandler(
            console=CONSOLE,
            show_level=False,
            show_path=False,
            show_time=False,
            rich_tracebacks=True,
        ),
        colorize=False,
        diagnose=True,
        backtrace=True,
        format=distributed.get_format(),
        filter=logging.OnceFilter(),
    )

    # ================================================================
    # Save the config.

    if config.load_config:
        config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)
        loguru.logger.success(f"📄 The config file <{config.load_config}> has been loaded.")

    config.set_timestamp()

    if distributed.is_master_rank():
        config.print_to_terminal()
        config.save_config()

    # ================================================================
    # Initialize the process group for distributed training.

    distributed.init_process_group(
        backend=config.distributed.backend,
        init_method=config.distributed.init_method,
        master_addr=config.distributed.master_addr,
        master_port=config.distributed.master_port,
        global_size=config.distributed.global_size,
        global_rank=config.distributed.global_rank,
        device_id=config.distributed.local_rank,
    )

    loguru.logger.success("🔥 Distributed process group has been initialized!")

    # ================================================================
    # Ensure that each process exclusively works on a single GPU.

    torch.cuda.set_device(config.distributed.local_rank)

    # ================================================================
    # Set the backend options.

    torch.backends.cudnn.benchmark = config.backend.benchmark
    torch.backends.cudnn.deterministic = config.backend.deterministic
    torch.use_deterministic_algorithms(config.backend.deterministic, warn_only=True)

    # ================================================================
    # Fix the random seed for reproducibility.

    seed = config.machine.seed + config.distributed.global_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ================================================================
    # Start training.

    trainer: CustomTrainer = config.setup(
        world_size=config.distributed.global_size,
        local_rank=config.distributed.local_rank,
    )
    trainer.setup()

    if distributed.is_master_rank():
        CONSOLE.rule("Model")
        CONSOLE.print(trainer.pipeline.model)
        CONSOLE.rule("")

    torch.distributed.barrier()

    loguru.logger.success("🚀 Training started!")

    trainer.train()

    torch.distributed.barrier()

    loguru.logger.success("✨ Training finished!")

    torch.distributed.destroy_process_group()


def entrypoint() -> None:
    tyro.extras.set_accent_color("bright_blue")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
