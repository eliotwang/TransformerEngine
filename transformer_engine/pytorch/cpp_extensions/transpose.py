# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for transpose extensions"""
<<<<<<< HEAD
from typing import Optional, Tuple, Union
import os
=======
from typing import List, Optional, Tuple, Union

>>>>>>> upstream/release_v1.11
import torch

import transformer_engine_torch as tex
from ..constants import TE_DType
<<<<<<< HEAD
from ..cast_transpose_triton import te_cast_transpose_noop_triton, te_cast_transpose_dbias_triton
=======
from ._common import canonicalize_fp8_scales, empty_tensor
>>>>>>> upstream/release_v1.11


__all__ = [
    "fp8_cast_transpose_fused",
    "fp8_cast_transpose_bgrad_fused",
    "fp8_cast_transpose_bgrad_dgelu_fused",
    "fp8_multi_cast_transpose_fused",
    "fp8_transpose_bgrad_fused",
]


def fp8_cast_transpose_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    cast_out: Optional[torch.Tensor] = None,
    transpose_out: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
    noop_flag: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cast + Transpose with FP8 output"""

    # Allocate outputs if needed
    if transpose_out is None:
        transpose_out = torch.empty(inp.shape[1], inp.shape[0], device="cuda", dtype=torch.uint8)
    if cast_out is None:
        cast_out = torch.empty_like(inp, dtype=torch.uint8)

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
    )

    # Construct no-op flag if needed
    if noop_flag is None:
        noop_flag = empty_tensor()

    # Launch kernel if needed
    if inp.nelement() > 0:
<<<<<<< HEAD
        use_cast_transpose_triton = bool( int(os.environ.get('NVTE_USE_CAST_TRANSPOSE_TRITON', '0')) )
        if use_cast_transpose_triton:
            te_cast_transpose_noop_triton(
                inp,
                noop_flag,
                fp8_meta_tensor.scale[fp8_tensor],
                cast_out,
                transpose_out,
                fp8_meta_tensor.amax_history[0][fp8_tensor],
                otype,
            )
        else:
            tex.fused_cast_transpose_noop(
                inp,
                noop_flag,
                fp8_meta_tensor.scale,
                fp8_meta_tensor.amax_history,
                fp8_meta_tensor.scale_inv,
                cast_out,
                transpose_out,
                otype,
                scale_offset=int(fp8_tensor),
                amax_offset=int(fp8_tensor),
                scale_inv_offset=int(fp8_tensor),
            )
=======
        tex.fused_cast_transpose_noop(
            inp,
            noop_flag,
            fp8_scales["scale"],
            fp8_scales["amax"],
            fp8_scales["scale_inv"],
            cast_out,
            transpose_out,
            otype,
            **fp8_scales_offsets,
        )
>>>>>>> upstream/release_v1.11

    return cast_out, transpose_out


def fp8_cast_transpose_bgrad_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast + Transpose + BGRAD with FP8 output"""
<<<<<<< HEAD
    use_cast_transpose_triton = bool( int(os.environ.get('NVTE_USE_CAST_TRANSPOSE_TRITON', '0')) )
    if use_cast_transpose_triton:
        return te_cast_transpose_dbias_triton(
            inp,
            fp8_meta_tensor.scale[fp8_tensor],
            fp8_meta_tensor.amax_history[0][fp8_tensor],
            otype,
        )
    else:
        return tex.fused_cast_transpose_bgrad(
            inp,
            fp8_meta_tensor.scale,
            fp8_meta_tensor.amax_history,
            fp8_meta_tensor.scale_inv,
            otype,
            scale_offset=int(fp8_tensor),
            amax_offset=int(fp8_tensor),
            scale_inv_offset=int(fp8_tensor),
        )
=======

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
    )

    # Launch kernel
    return tex.fused_cast_transpose_bgrad(
        inp,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        otype,
        **fp8_scales_offsets,
    )
>>>>>>> upstream/release_v1.11


def fp8_transpose_bgrad_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    grad_bias_type: torch.dtype,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transpose + BGRAD with FP8 output"""

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
    )

    # Launch kernel
    return tex.fused_fp8_transpose_bgrad(
        inp,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        otype,
        TE_DType[grad_bias_type],
        **fp8_scales_offsets,
    )


def fp8_cast_transpose_bgrad_dgelu_fused(
    grad_output: torch.Tensor,
    gelu_input: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast + Transpose + BGRAD + DGELU with FP8 output"""

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
    )

    # Launch kernel
    return tex.fused_cast_transpose_bgrad_dgelu(
        grad_output,
        gelu_input,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        otype,
        **fp8_scales_offsets,
    )


def fp8_multi_cast_transpose_fused(
    input_list: List[torch.Tensor],
    fp8_meta_tensor: tex.FP8TensorMeta,
    scale_indices: List[int],
    amax_indices: List[int],
    scale_inv_indices: List[int],
    otype: tex.DType,
    scale_inv: Optional[torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Cast + Transpose with FP8 output"""

    return tex.fused_multi_cast_transpose_alloc(
        input_list,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        scale_inv if scale_inv is not None else fp8_meta_tensor.scale_inv,
        scale_indices,
        amax_indices,
        scale_inv_indices,
        otype,
    )
