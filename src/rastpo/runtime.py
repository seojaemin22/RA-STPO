"""Explicit devices, reproducible seeds, and measured resource records."""

from __future__ import annotations

from contextlib import contextmanager
import platform
from pathlib import Path
import resource
import time

import numpy as np
import torch


def device_from_name(name: str) -> torch.device:
    device = torch.device(
        "cuda"
        if name == "auto" and torch.cuda.is_available()
        else "cpu" if name == "auto" else name
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def environment(device) -> dict:
    device = torch.device(device)
    cpu_model = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return dict(
        cpu_model=cpu_model,
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=str(device),
        threads=torch.get_num_threads(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        gpu_capability=(
            list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None
        ),
        gpu_multiprocessors=(
            torch.cuda.get_device_properties(device).multi_processor_count
            if device.type == "cuda"
            else None
        ),
        accelerator=(
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
    )


@contextmanager
def measured(device, *, reset_peak=True):
    """Elapsed wall time and process-level high-water marks, not sampled estimates.

    CPU RSS is the process lifetime maximum. CUDA allocated/reserved peaks cover
    this interval when reset_peak is True. These quantities are not additive.
    """
    device = torch.device(device)
    synchronize(device)
    if reset_peak and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        synchronize(device)
        result.update(
            seconds=time.perf_counter() - start,
            cpu_process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024**2 if platform.system() == "Darwin" else 1024),
            gpu_peak_allocated_mib=(
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
            gpu_peak_reserved_mib=(
                torch.cuda.max_memory_reserved(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
        )
