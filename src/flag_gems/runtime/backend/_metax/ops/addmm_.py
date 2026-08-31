# Copyright 2026 FlagOS Contributors
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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _copy_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def _addmm_kernel(
    self_ptr, mat1_ptr, mat2_ptr,
    M, N, K,
    stride_sm, stride_sn,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    alpha, beta,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
    TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    if EVEN_M:
        m_mask = tl.full((BLOCK_M, 1), 1, tl.int1)
    else:
        m_mask = (offs_m < M)[:, None]
    if EVEN_N:
        n_mask = tl.full((1, BLOCK_N), 1, tl.int1)
    else:
        n_mask = (offs_n < N)[None, :]
    mn_mask = m_mask & n_mask

    a_ptrs = mat1_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = mat2_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if alpha != 0.0:
        if EVEN_K:
            for _ in range(0, K, BLOCK_K):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                if TF32:
                    acc = tl.dot(a, b, acc, input_precision="tf32")
                else:
                    acc = tl.dot(a, b, acc, input_precision="ieee")
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        else:
            for _ in range(0, tl.cdiv(K, BLOCK_K)):
                kk = offs_k < (K - _ * BLOCK_K)
                a = tl.load(a_ptrs, mask=m_mask & kk[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=kk[:, None] & n_mask, other=0.0)
                if TF32:
                    acc = tl.dot(a, b, acc, input_precision="tf32")
                else:
                    acc = tl.dot(a, b, acc, input_precision="ieee")
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk

    s_ptrs = self_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    if beta != 0.0:
        s = tl.load(s_ptrs, mask=mn_mask, other=0.0)
        out = alpha * acc + beta * s
    else:
        out = alpha * acc

    out = out.to(self_ptr.dtype.element_ty)
    tl.store(s_ptrs, out, mask=mn_mask)


_GROUP_M = 8
_NUM_STAGES = 2


def _pick_config(M, N, K, dtype):
    itemsize = dtype.itemsize
    if itemsize <= 2 and max(M, N) > 1536:
        if (M % 256 == 0) and (N % 256 == 0) and (K % 32 == 0):
            if dtype == torch.float16 or min(M, N) >= 2560:
                return 256, 256, 32, 8
        return 128, 128, 64, 8
    if itemsize <= 2:
        if max(M, N) <= 512 and (M % 32 == 0) and (N % 32 == 0) and (K % 64 == 0):
            return 32, 32, 64, 4
        return 64, 64, 64, 4
    if min(M, N) >= 2560:
        return 128, 128, 32, 8
    if M % 128 == 0 and N % 128 == 0 and K >= 1024 and K % 32 == 0:
        return 128, 128, 32, 8
    if max(M, N) <= 512 and (M % 32 == 0) and (N % 32 == 0):
        return 32, 32, 32, 4
    return 64, 64, 32, 8


def _snapshot_copy(t, device):
    out = torch.empty_like(t)
    numel = t.numel()
    block = 1024
    grid = (triton.cdiv(numel, block),)
    with torch_device_fn.device(device):
        _copy_kernel[grid](t, out, numel, BLOCK=block, num_warps=4)
    return out


def addmm_(self, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS_METAX ADDMM_")
    assert self.dtype.is_floating_point, "Only floating-point dtypes are supported"
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        self.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"

    M, K = mat1.shape
    N = mat2.shape[1]
    if M == 0 or N == 0:
        return self

    alpha_f = float(alpha)
    beta_f = float(beta)

    # In-place aliasing guard: snapshot any operand that shares storage with
    # self so concurrent Triton programs never observe half-written tiles.
    if self.data_ptr() == mat1.data_ptr():
        mat1 = _snapshot_copy(mat1, self.device)
    if self.data_ptr() == mat2.data_ptr():
        mat2 = _snapshot_copy(mat2, self.device)

    # MetaX lowers GEMM load efficiently when mat1 is row-major and mat2 is
    # column-contiguous in N; mirror the contiguity fix from the addmm path.
    if mat1.stride(0) > 1 and mat1.stride(1) > 1:
        mat1 = mat1.contiguous()
    if mat2.stride(1) != 1:
        mat2 = mat2.contiguous()

    stride_sm, stride_sn = self.stride()
    stride_am, stride_ak = mat1.stride()
    stride_bk, stride_bn = mat2.stride()

    block_m, block_n, block_k, num_warps = _pick_config(M, N, K, self.dtype)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)

    extra = {}
    if self.dtype.itemsize <= 2:
        if block_m == 256 and block_n == 256:
            extra["scenario"] = "roll"
        elif block_m == block_n and block_m in (64, 32) and K % block_k == 0:
            extra["scenario"] = "roll"
    elif block_m == 64 and block_n == 64 and K >= 1024 and (M % block_m == 0) and (N % block_n == 0):
        extra["scenario"] = "roll"
    elif block_m == 128 and block_n == 128 and K >= 1024 and (M % block_m == 0) and (N % block_n == 0):
        extra["scenario"] = "roll"

    use_tf32 = (
        self.dtype == torch.float32
        and K >= 1024
        and (M % 64 == 0)
        and (N % 64 == 0)
        and (K % 32 == 0)
    )

    with torch_device_fn.device(self.device):
        _addmm_kernel[grid](
            self, mat1, mat2,
            M, N, K,
            stride_sm, stride_sn,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            alpha_f, beta_f,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            GROUP_M=_GROUP_M,
            EVEN_M=(M % block_m == 0),
            EVEN_N=(N % block_n == 0),
            EVEN_K=(K % block_k == 0),
            TF32=use_tf32,
            num_warps=num_warps,
            num_stages=_NUM_STAGES,
            **extra,
        )
    return self
