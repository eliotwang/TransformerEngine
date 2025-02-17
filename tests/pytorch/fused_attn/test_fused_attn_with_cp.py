# This file was modified for portability to AMDGPU
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import pytest
import subprocess
from torch.cuda import device_count
from torch.utils.cpp_extension import IS_HIP_EXTENSION
from test_fused_attn import ModelConfig
from transformer_engine.pytorch.attention import (
    _flash_attn_2_plus,
    _flash_attn_2_3_plus,
)
from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
    get_cudnn_version,
)

model_configs_flash_attn = {
    #   test:             b,  h, hg,   d,   sq,  skv,   p,      mask,      bias
    "cp_1_0": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "causal", "no_bias"),  # MHA
    "cp_1_1": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "no_mask", "no_bias"),  # MHA
    "cp_2_0": ModelConfig(2, 12, 1, 128, 4096, 4096, 0.0, "causal", "no_bias"),  # GQA
    "cp_2_1": ModelConfig(2, 12, 1, 128, 4096, 4096, 0.0, "no_mask", "no_bias"),  # GQA
}


def get_bash_arguments(**kwargs):
    args = ["python", "-m", "torch.distributed.launch", "--nproc-per-node=2"]
    te_path = os.getenv("TE_PATH", "/opt/transformerengine")
    script_path = os.path.join(te_path, "tests/pytorch/fused_attn/run_fused_attn_with_cp.py")
    args.append(script_path)
    for k, v in kwargs.items():
        args.append(f"{k}={v}")
    return args


@pytest.mark.skipif(not _flash_attn_2_plus, reason="Flash-attn 2.0+ is required.")
@pytest.mark.skipif(not IS_HIP_EXTENSION and get_device_compute_capability() < (8, 0), reason="CP tests require sm80+.")
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("model", model_configs_flash_attn.keys())
@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd", "thd"])
@pytest.mark.skipif(device_count() < 2, reason="multi-GPU host is required")
def test_cp_with_flash_attention(dtype, model, qkv_format):
    subprocess.run(
        get_bash_arguments(
            dtype=dtype, model=model, qkv_format=qkv_format, kernel_backend="FlashAttention"
        ),
        check=True,
    )

#TODO: release bias tests once CK/AOTriton support bias
if IS_HIP_EXTENSION:
    model_configs_fused_attn = {
        #   test:             b,  h, hg,   d,    sq,   skv,   p,      mask,      bias
        "cp_1_0": ModelConfig(2, 12, 12, 128,  4096,  4096, 0.0,  "causal", "no_bias"),  # MHA
        "cp_1_1": ModelConfig(2, 12, 12, 128,  4096,  4096, 0.0, "no_mask", "no_bias"),  # MHA
        "cp_2_0": ModelConfig(2, 12, 1, 128, 4096, 4096, 0.0, "causal", "no_bias"),  # GQA
        "cp_2_1": ModelConfig(2, 12, 1, 128, 4096, 4096, 0.0, "no_mask", "no_bias"),  # GQA
    }
else:
    model_configs_fused_attn = {
        #   test:             b,  h, hg,   d,   sq,  skv,   p,      mask,              bias
        "cp_1_0": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "causal", "no_bias"),  # MHA
        "cp_1_1": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "no_mask", "no_bias"),  # MHA
        "cp_1_2": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "causal", "post_scale_bias"),  # MHA
        "cp_1_3": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "no_mask", "post_scale_bias"),  # MHA
        "cp_2_0": ModelConfig(2, 12, 1, 128, 4096, 4096, 0.0, "causal", "no_bias"),  # GQA
        "cp_2_1": ModelConfig(2, 12, 1, 128, 4096, 4096, 0.0, "no_mask", "no_bias"),  # GQA
        "cp_2_2": ModelConfig(2, 12, 1, 128, 4096, 4096, 0.0, "causal", "post_scale_bias"),  # GQA
        "cp_2_3": ModelConfig(2, 12, 1, 128, 4096, 4096, 0.0, "no_mask", "post_scale_bias"),  # GQA
    }


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 7), reason="cuDNN 8.9.7+ is required.")
@pytest.mark.skipif(not IS_HIP_EXTENSION and get_device_compute_capability() < (8, 0), reason="CP tests require sm80+.")
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("model", model_configs_fused_attn.keys())
@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd"] if IS_HIP_EXTENSION else ["bshd", "sbhd", "thd"])
@pytest.mark.skipif(device_count() < 2, reason="multi-GPU host is required")
def test_cp_with_fused_attention(dtype, model, qkv_format):
    if qkv_format == "thd" and get_device_compute_capability() < (9, 0):
        pytest.skip("THD format is only supported on sm90+.")
    subprocess.run(
        get_bash_arguments(
            dtype=dtype, model=model, qkv_format=qkv_format, kernel_backend="FusedAttention"
        ),
        check=True,
    )
