from functools import lru_cache
import logging
from typing import Any, Dict, List, MutableSequence, Optional
from typing_extensions import override

import torch

from lightning.fabric.utilities.distributed import _distributed_is_initialized
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.warnings import PossibleUserWarning


import lightning.pytorch as pl
from lightning.pytorch.accelerators import (
    CUDAAccelerator,
    MPSAccelerator,
    XLAAccelerator,
)
from lightning.pytorch.plugins.precision import MixedPrecision
from lightning.pytorch.plugins.precision.precision import Precision
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy, Strategy
from lightning.pytorch.strategies.single_xla import SingleDeviceXLAStrategy
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _AcceleratorConnector,
    _IS_INTERACTIVE,
)
from lightning.pytorch.trainer import setup
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn
from torch.nn.parallel import DistributedDataParallel


_log = logging.getLogger(__name__)

"""
Module that defines XPUAccelerator: Accelerator subclass for Intel GPUs (XPUs).

The class XPUAccelerator has been adapted from the PyTorch Lightning class:
- lightning.pytorch.accelerators.cuda.CUDAAccelerator

PyTorch Lightning is licensed under version 2.0 of the Apache License:
- https://www.apache.org/licenses/LICENSE-2.0.html
"""


class XPUAccelerator(Accelerator):
    """Accelerator for Intel GPUs (XPUs)."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not an XPU.
        """
        if device.type != "xpu":
            raise MisconfigurationException(
                f"Device should be XPU, got {device} instead"
            )
        torch.xpu.set_device(device)

        # Set default values for environment variables
        # relevant when using XPU devices in parallel processing.
        # os.environ.setdefault("CCL_WORKER_OFFLOAD", "0")
        # https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-9/environment-variables.html
        # os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")
        # https://uxlfoundation.github.io/oneCCL/env-variables.html
        # os.environ.setdefault("CCL_ZE_IPC_EXCHANGE", "pidfd")
        # https://www.intel.com/content/www/us/en/developer/articles/technical/flattening-gpu-tile-hierarchy.html
        # os.environ.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
        # mask = ",".join(str(idx) for idx in _get_all_visible_xpu_devices())
        # os.environ.setdefault("ZE_AFFINITY_MASK", mask)

    @override
    def get_device_stats(self, device: _DEVICE) -> Dict[str, Any]:
        """Gets stats for the given GPU device.

        Args:
            device: GPU device for which to get stats

        Returns:
            A dictionary mapping the metrics to their values.

        Raises:
            FileNotFoundError:
                If xpu-smi installation not found
        """
        return torch.xpu.memory_stats(device)

    @override
    def teardown(self) -> None:
        torch.xpu.empty_cache()

    # Parsing code largely adapted from:
    # lightning.pytorch.utilities.device_parser._parse_gpu_ids(),
    # and functions called from there.
    @staticmethod
    @override
    def parse_devices(devices: Any) -> Any:
        """Accelerator device parsing logic."""
        from lightning.fabric.utilities.device_parser import (
            _check_data_type,
            _check_unique,
            _normalize_parse_gpu_string_input,
        )

        # Check that devices param is None, Int, String or Sequence of Ints
        _check_data_type(devices)

        # Handle the case when no GPUs are requested
        if (
            devices is None
            or (isinstance(devices, int) and devices == 0)
            or str(devices).strip() in ("0", "[]")
        ):
            return None

        # If requested GPUs are not available, throw an exception.
        gpus = _normalize_parse_gpu_string_input(devices)
        if isinstance(gpus, (MutableSequence, tuple)):
            gpus = list(gpus)

        if not gpus:
            raise MisconfigurationException("GPUs requested but none available.")

        all_available_gpus = _get_all_visible_xpu_devices()
        if -1 == gpus:
            return all_available_gpus
        elif isinstance(gpus, int):
            gpus = list(range(gpus))

        # Check that GPUs are unique.
        # Duplicate GPUs are not supported by the backend.
        _check_unique(gpus)

        for gpu in gpus:
            if gpu not in all_available_gpus:
                raise MisconfigurationException(
                    f"You requested gpu: {gpus}\n"
                    f"But your machine only has: {all_available_gpus}"
                )

        return gpus

    @override
    @staticmethod
    def get_parallel_devices(devices: Any) -> Any:
        """Gets parallel devices for the Accelerator."""
        if isinstance(devices, int):
            devices = range(devices)
        return [torch.device("xpu", idx) for idx in devices]

    @override
    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return num_xpu_devices()

    @override
    @staticmethod
    def is_available() -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    @staticmethod
    @override
    def name() -> str:
        return "xpu"

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: Any) -> None:
        accelerator_registry.register(
            cls.name(),
            cls,
            description=cls.__class__.__name__,
        )


@lru_cache(1)
def num_xpu_devices() -> int:
    """Return the number of available XPU devices."""
    if not hasattr(torch, "xpu"):
        return 0
    return torch.xpu.device_count()


def _get_all_visible_xpu_devices() -> List[int]:
    """
    Return a list of all visible XPU devices.
    The devices returned depend on the values set for the environment
    variables ``ZE_FLAT_DEVICE_HIERARCHY`` and ``ZE_AFFINITY_MASK``
    """
    return list(range(num_xpu_devices()))


# From the trainer setup:
#
# Modifications to lightning.pytorch.trainer.setup._log_device_info()
#


def _xpu_log_device_info(trainer: "pl.Trainer") -> None:
    if CUDAAccelerator.is_available():
        gpu_available = True
        gpu_type = " (cuda)"
    elif MPSAccelerator.is_available():
        gpu_available = True
        gpu_type = " (mps)"
    elif XPUAccelerator.is_available():
        gpu_available = True
        gpu_type = " (xpu)"
    else:
        gpu_available = False
        gpu_type = ""

    gpu_used = isinstance(
        trainer.accelerator, (CUDAAccelerator, MPSAccelerator, XPUAccelerator)
    )
    num_gpus = trainer.num_devices if gpu_used else 0
    rank_zero_info(
        f"GPU available: {gpu_available}{gpu_type}, using: "
        f"{num_gpus} {'GPU' if 1 == num_gpus else 'GPUs'}{gpu_type}"
    )

    num_tpu_cores = (
        trainer.num_devices if isinstance(trainer.accelerator, XLAAccelerator) else 0
    )
    rank_zero_info(
        f"TPU available: {XLAAccelerator.is_available()}, using: {num_tpu_cores} TPU cores"
    )

    num_hpus = 0
    hpu_available = False
    rank_zero_info(f"HPU available: {hpu_available}, using: {num_hpus} HPUs")

    if (
        CUDAAccelerator.is_available()
        and not isinstance(trainer.accelerator, CUDAAccelerator)
        or MPSAccelerator.is_available()
        and not isinstance(trainer.accelerator, MPSAccelerator)
        or XPUAccelerator.is_available()
        and not isinstance(trainer.accelerator, XPUAccelerator)
    ):
        rank_zero_warn(
            "GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.",
            category=PossibleUserWarning,
        )

    if XLAAccelerator.is_available() and not isinstance(
        trainer.accelerator, XLAAccelerator
    ):
        rank_zero_warn(
            "TPU available but not used. You can set it by doing `Trainer(accelerator='tpu')`."
        )


def _xpu_choose_strategy(self: "_AcceleratorConnector") -> "Strategy | str":
    """Patch Lightning's single-device strategy selection to handle XPU."""
    if self._accelerator_flag == "hpu":
        raise MisconfigurationException(
            "HPU is currently not supported. Please contact developer@lightning.ai"
        )

    if self._accelerator_flag == "tpu" or isinstance(
        self._accelerator_flag, XLAAccelerator
    ):
        if self._parallel_devices and len(self._parallel_devices) > 1:
            return "xla"
        return SingleDeviceXLAStrategy(device=self._parallel_devices[0])

    if self._num_nodes_flag > 1:
        return "ddp"

    if len(self._parallel_devices) <= 1:
        if isinstance(
            self._accelerator_flag, (CUDAAccelerator, MPSAccelerator, XPUAccelerator)
        ) or (
            isinstance(self._accelerator_flag, str)
            and self._accelerator_flag in ("cuda", "gpu", "mps", "xpu")
        ):
            device = self._parallel_devices[0]
        else:
            device = "cpu"
        return SingleDeviceStrategy(device=device)

    if len(self._parallel_devices) > 1 and _IS_INTERACTIVE:
        return "ddp_fork"
    return "ddp"


setup._log_device_info = _xpu_log_device_info
_AcceleratorConnector._choose_strategy = _xpu_choose_strategy


# Patch Lightning's precision selection so mixed precision on XPU uses xpu autocast.
_original_check_and_init_precision = _AcceleratorConnector._check_and_init_precision


def _xpu_check_and_init_precision(self: "_AcceleratorConnector"):
    self._validate_precision_choice()

    if isinstance(self._precision_plugin_flag, Precision):
        return self._precision_plugin_flag

    is_xpu = isinstance(self._accelerator_flag, XPUAccelerator) or (
        isinstance(self._accelerator_flag, str) and self._accelerator_flag == "xpu"
    )
    if is_xpu and self._precision_flag in ("16-mixed", "bf16-mixed"):
        rank_zero_info(
            f"Using {'16bit' if self._precision_flag == '16-mixed' else 'bfloat16'} "
            "Automatic Mixed Precision (AMP) on XPU"
        )
        return MixedPrecision(self._precision_flag, "xpu")

    return _original_check_and_init_precision(self)


_AcceleratorConnector._check_and_init_precision = _xpu_check_and_init_precision


# Patch Lightning's DDPStrategy to avoid CUDA-only stream context on XPU.
def _xpu_ddp_setup_model(
    self: "DDPStrategy", model: torch.nn.Module
) -> DistributedDataParallel:
    """Wrap the model in DDP without forcing non-default streams for XPU devices."""
    device_ids = self.determine_ddp_device_ids()
    _log.debug(
        "setting up DDP model with device ids: %s, kwargs: %s",
        device_ids,
        self._ddp_kwargs,
    )
    # Keep CUDA behavior (Lightning default) but avoid a custom XPU stream context.
    if device_ids is not None and getattr(self.root_device, "type", None) == "cuda":
        with torch.cuda.stream(torch.cuda.Stream()):
            return DistributedDataParallel(
                module=model, device_ids=device_ids, **self._ddp_kwargs
            )

    return DistributedDataParallel(
        module=model, device_ids=device_ids, **self._ddp_kwargs
    )


DDPStrategy._setup_model = _xpu_ddp_setup_model


def _xpu_barrier(self, name: Optional[str] = None) -> None:
    if not _distributed_is_initialized():
        return
    if torch.distributed.get_backend() in ["nccl", "xccl", "ccl"]:
        torch.distributed.barrier(device_ids=self.determine_ddp_device_ids())
    else:
        torch.distributed.barrier()


DDPStrategy.barrier = _xpu_barrier
