# Copyright 2026 FlagOS Contributors
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strict Ascend kernel timing for the FlagGems benchmark suite."""

from __future__ import annotations

import csv
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from typing import Any

import triton
import triton.language as tl

L2_CACHE_SIZE_DEFAULT = 192 * 1024 * 1024
L2_CACHE_CLEAR_KERNEL_NAME = "FlagGems_l2cache_clear"
_MAX_WARMUP_COUNT = 100_000
_MAX_ACTIVE_COUNT = 100_000
_L2_CACHE_BUFFERS: dict[int, Any] = {}
_L2_CACHE_SIZES: dict[int, int] = {}
_CORE_COUNTS: dict[int, int] = {}


@triton.jit
def FlagGems_l2cache_clear(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    """Overwrite one L2-sized buffer using a uniquely named Triton kernel."""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_idx in range(pid, num_blocks, CORE_NUM):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(
            output_ptr + offsets,
            tl.zeros([BLOCK_SIZE], dtype=tl.int32),
            mask=offsets < n_elements,
        )


def _torch_modules():
    # Importing torch_npu acquires Ascend runtime state. Keep it out of benchmark
    # collection and all non-Ascend processes.
    import torch
    import torch_npu

    return torch, torch_npu


def _current_device() -> int:
    torch, _ = _torch_modules()
    return int(torch.npu.current_device())


def _l2_cache_size(device_id: int) -> int:
    cached = _L2_CACHE_SIZES.get(device_id)
    if cached is not None:
        return cached

    _, torch_npu = _torch_modules()
    size = L2_CACHE_SIZE_DEFAULT
    try:
        properties = torch_npu.npu.get_device_properties(device_id)
        detected = int(getattr(properties, "L2_cache_size", 0) or 0)
        if detected > 0:
            size = detected
    except Exception:
        pass
    _L2_CACHE_SIZES[device_id] = size
    return size


def _vector_core_count(device_id: int) -> int:
    cached = _CORE_COUNTS.get(device_id)
    if cached is not None:
        return cached

    count = 40
    try:
        properties = triton.runtime.driver.active.utils.get_device_properties(device_id)
        count = int(properties.get("num_vectorcore", count) or count)
    except Exception:
        pass
    _CORE_COUNTS[device_id] = count
    return count


def _l2_cache_buffer(device_id: int):
    buffer = _L2_CACHE_BUFFERS.get(device_id)
    if buffer is None:
        torch, _ = _torch_modules()
        element_count = max(1, _l2_cache_size(device_id) // 4)
        buffer = torch.empty(
            element_count,
            dtype=torch.int32,
            device=f"npu:{device_id}",
        )
        _L2_CACHE_BUFFERS[device_id] = buffer
    return buffer


def clear_l2_cache() -> None:
    """Evict Ascend L2 contents and wait until the eviction kernel completes."""
    torch, _ = _torch_modules()
    device_id = _current_device()
    buffer = _l2_cache_buffer(device_id)
    core_count = _vector_core_count(device_id)
    block_size = 32768
    FlagGems_l2cache_clear[(core_count,)](
        buffer,
        buffer.numel(),
        BLOCK_SIZE=block_size,
        CORE_NUM=core_count,
    )
    torch.npu.synchronize()


def _budget_to_counts(
    fn: Callable[[], Any],
    warmup_ms: float,
    repetition_ms: float,
    probe_runs: int = 5,
) -> tuple[int, int]:
    """Convert millisecond budgets to bounded profiler iteration counts."""
    if warmup_ms < 0 or repetition_ms <= 0:
        raise ValueError(
            "Ascend timing requires warmup_ms >= 0 and repetition_ms > 0, got "
            f"{warmup_ms!r} and {repetition_ms!r}"
        )
    if probe_runs < 1:
        raise ValueError("probe_runs must be positive")

    torch, _ = _torch_modules()
    for _ in range(2):
        fn()
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(probe_runs):
        fn()
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    latency_ms = max(elapsed_ms / probe_runs, 0.001)

    warmup_count = max(0, int(warmup_ms / latency_ms))
    active_count = max(1, int(repetition_ms / latency_ms))
    return (
        min(warmup_count, _MAX_WARMUP_COUNT),
        min(active_count, _MAX_ACTIVE_COUNT),
    )


def _is_l2_clear_row(row: dict[str, str]) -> bool:
    name = row.get("OP Type", row.get("Name", ""))
    return L2_CACHE_CLEAR_KERNEL_NAME.lower() in str(name).lower()


def _collect_latency_us(profile_dir: str, active_count: int) -> float:
    """Return the summed device-kernel duration per measured invocation."""
    for root, _, files in os.walk(profile_dir):
        for filename in sorted(files):
            if filename != "op_statistic.csv" and not filename.startswith(
                "op_statistic"
            ):
                continue

            csv_path = os.path.join(root, filename)
            try:
                with open(csv_path, newline="", encoding="utf-8-sig") as handle:
                    rows = list(csv.DictReader(handle))
            except (csv.Error, OSError):
                continue
            rows = [row for row in rows if not _is_l2_clear_row(row)]
            if not rows:
                continue

            columns = set(rows[0])
            if not {"Count", "Total Time(us)"}.issubset(columns):
                continue
            try:
                measured_rows = [
                    row
                    for row in rows
                    if int(float(row["Count"])) % active_count == 0
                ]
                total_us = sum(float(row["Total Time(us)"]) for row in measured_rows)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            if measured_rows and math.isfinite(total_us) and total_us > 0:
                return total_us / active_count

    raise RuntimeError(
        "Ascend profiler did not produce valid op_statistic.csv timing data"
    )


def _profile_latency_us(
    fn: Callable[[], Any],
    warmup_count: int,
    active_count: int,
    clear_cache: bool,
    keep_profile: bool,
) -> float:
    torch, torch_npu = _torch_modules()
    profile_dir = tempfile.mkdtemp(
        prefix=".flaggems_ascend_profile_", dir=os.getcwd()
    )
    try:
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=False,
        )
        skip_first = 1 + warmup_count
        total_count = skip_first + active_count

        # Compile the cache-clear kernel before entering the measured profiler window.
        if clear_cache:
            clear_l2_cache()

        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(
                wait=0,
                warmup=0,
                active=active_count,
                repeat=1,
                skip_first=skip_first,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_dir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config,
        ) as profiler:
            for _ in range(total_count):
                if clear_cache:
                    clear_l2_cache()
                fn()
                profiler.step()
                torch.npu.synchronize()

        return _collect_latency_us(profile_dir, active_count)
    finally:
        if not keep_profile:
            shutil.rmtree(profile_dir, ignore_errors=True)


def measure_latency(
    fn: Callable[[], Any],
    warmup_ms: float,
    repetition_ms: float,
    *,
    clear_l2_cache: bool = True,
    keep_profile: bool = False,
) -> float:
    """Measure one Ascend invocation and return strict profiler time in ms.

    ``warmup_ms`` and ``repetition_ms`` remain time budgets, matching the public
    FlagGems benchmark CLI. They are calibrated into iteration counts before
    entering ``torch_npu.profiler``. No wall-time fallback is permitted.
    """
    torch, _ = _torch_modules()

    # Keep JIT compilation and lazy runtime initialization outside calibration and
    # the profiler window.
    fn()
    torch.npu.synchronize()
    warmup_count, active_count = _budget_to_counts(
        fn,
        warmup_ms=warmup_ms,
        repetition_ms=repetition_ms,
    )
    latency_us = _profile_latency_us(
        fn,
        warmup_count=warmup_count,
        active_count=active_count,
        clear_cache=clear_l2_cache,
        keep_profile=keep_profile,
    )
    if not math.isfinite(latency_us) or latency_us <= 0:
        raise RuntimeError(
            f"Ascend profiler produced invalid timing: {latency_us!r} us"
        )
    return latency_us / 1000.0
