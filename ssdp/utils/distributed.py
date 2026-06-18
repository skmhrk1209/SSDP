from __future__ import annotations

import contextlib
import datetime
import enum
import functools
import os
import socket
import types
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru._defaults import LOGURU_FORMAT

if TYPE_CHECKING:
    from mpi4py import MPI


class Backend(enum.StrEnum):
    GLOO = torch.distributed.Backend.GLOO
    NCCL = torch.distributed.Backend.NCCL
    XCCL = torch.distributed.Backend.XCCL
    UCC = torch.distributed.Backend.UCC
    MPI = torch.distributed.Backend.MPI


class InitMethod(enum.StrEnum):
    ENV = enum.auto()
    TCP = enum.auto()
    FILE = enum.auto()


class OpenMPIEnvKey(enum.StrEnum):
    GLOBAL_SIZE = "OMPI_COMM_WORLD_SIZE"
    GLOBAL_RANK = "OMPI_COMM_WORLD_RANK"
    LOCAL_SIZE = "OMPI_COMM_WORLD_LOCAL_SIZE"
    LOCAL_RANK = "OMPI_COMM_WORLD_LOCAL_RANK"


class IntelMPIEnvKey(enum.StrEnum):
    GLOBAL_SIZE = "PMI_SIZE"
    GLOBAL_RANK = "PMI_RANK"
    LOCAL_SIZE = "MPI_LOCALNRANKS"
    LOCAL_RANK = "MPI_LOCALRANKID"


def init_process_group(
    backend: str = Backend.NCCL,
    init_method: str = InitMethod.ENV,
    master_addr: str | None = None,
    master_port: int | None = None,
    global_size: int | None = None,
    global_rank: int | None = None,
    shared_file: Path | None = None,
    device_id: torch.device | int | None = None,
    timeout: datetime.timedelta | None = None,
) -> None:
    backend = Backend(backend)
    init_method = InitMethod(init_method)

    if init_method is not InitMethod.FILE:
        if master_addr is None:
            master_addr = get_master_address()

        if master_port is None:
            master_port = get_master_port()

    if init_method is InitMethod.ENV:
        os.environ.update(
            MASTER_ADDR=master_addr,
            MASTER_PORT=str(master_port),
        )
        init_method = f"{init_method}://"

    elif init_method is InitMethod.TCP:
        init_method = f"{init_method}://{master_addr}:{master_port}"

    elif init_method is InitMethod.FILE:
        if shared_file is None:
            raise ValueError(
                f"`shared_file` must be specified when `init_method` is {InitMethod.FILE!r}."
            )
        if shared_file.exists():
            raise FileExistsError(
                f"The shared file <{shared_file}> already exists. "
                f"Please specify a non-existent file."
            )
        if not shared_file.parent.exists():
            raise FileNotFoundError(
                f"The parent directory of the shared file <{shared_file}> does not exist."
            )
        init_method = f"{init_method}://{shared_file}"

    else:
        raise ValueError(
            f"Invalid value {init_method!r} for `init_method`. "
            f"Only {InitMethod.ENV!r}, {InitMethod.TCP!r}, and {InitMethod.FILE!r} are supported."
        )

    if global_size is None:
        global_size = get_global_size()

    if global_rank is None:
        global_rank = get_global_rank()

    if device_id is None:
        device_id = get_local_rank()

    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=global_size,
        rank=global_rank,
        device_id=device_id,
        timeout=timeout,
    )


@functools.cache
def _mpi() -> types.ModuleType:
    try:
        from mpi4py import MPI

        if not MPI.Is_initialized():
            raise RuntimeError("`mpi4py` has not been initialized.")
    except Exception as error:
        raise RuntimeError("`mpi4py` must be available if Open MPI is unavailable.") from error
    return MPI


@functools.cache
def _get_global_comm() -> MPI.Comm:
    return _mpi().COMM_WORLD


@functools.cache
def _get_local_comm() -> MPI.Comm:
    global_comm = _get_global_comm()
    local_comm = global_comm.Split_type(_mpi().COMM_TYPE_SHARED)
    return local_comm


def _get_ipv4_address() -> str:
    hostname = socket.gethostname()
    ipv4_address = socket.gethostbyname(hostname)
    return ipv4_address


@functools.cache
def get_master_address(master_rank: int = 0) -> str:
    ipv4_address = _get_ipv4_address()
    global_comm = _get_global_comm()
    master_address = global_comm.bcast(ipv4_address, master_rank)
    return master_address


def _get_free_port() -> int:
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(("", 0))
        _, free_port = sock.getsockname()
    return free_port


@functools.cache
def get_master_port(master_rank: int = 0) -> int:
    free_port = _get_free_port()
    global_comm = _get_global_comm()
    master_port = global_comm.bcast(free_port, master_rank)
    return master_port


def get_global_size() -> int:
    if torch.distributed.is_initialized():
        global_size = torch.distributed.get_world_size()
    else:
        for key in (OpenMPIEnvKey.GLOBAL_SIZE, IntelMPIEnvKey.GLOBAL_SIZE):
            if (global_size := os.environ.get(key)) is not None:
                break
        else:
            global_comm = _get_global_comm()
            global_size = global_comm.Get_size()
    return int(global_size)


def get_global_rank() -> int:
    if torch.distributed.is_initialized():
        global_rank = torch.distributed.get_rank()
    else:
        for key in (OpenMPIEnvKey.GLOBAL_RANK, IntelMPIEnvKey.GLOBAL_RANK):
            if (global_rank := os.environ.get(key)) is not None:
                break
        else:
            global_comm = _get_global_comm()
            global_rank = global_comm.Get_rank()
    return int(global_rank)


def get_local_size() -> int:
    for key in (OpenMPIEnvKey.LOCAL_SIZE, IntelMPIEnvKey.LOCAL_SIZE):
        if (local_size := os.environ.get(key)) is not None:
            break
    else:
        local_comm = _get_local_comm()
        local_size = local_comm.Get_size()
    return int(local_size)


def get_local_rank() -> int:
    for key in (OpenMPIEnvKey.LOCAL_RANK, IntelMPIEnvKey.LOCAL_RANK):
        if (local_rank := os.environ.get(key)) is not None:
            break
    else:
        local_comm = _get_local_comm()
        local_rank = local_comm.Get_rank()
    return int(local_rank)


def is_master_rank(master_rank: int = 0) -> bool:
    global_rank = get_global_rank()
    return global_rank == master_rank


def get_format(base_format: str = LOGURU_FORMAT) -> str:
    prefix, suffix = base_format.rsplit(" | ", 1)
    global_size = get_global_size()
    global_rank = get_global_rank()
    local_size = get_local_size()
    local_rank = get_local_rank()
    global_info = f"<level>Global Rank: [{global_rank}/{global_size}]</level>"
    local_info = f"<level>Local Rank: [{local_rank}/{local_size}]</level>"
    format = " | ".join([prefix, global_info, local_info, suffix])
    return f"{format}\n"
