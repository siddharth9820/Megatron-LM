# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import torch

# from .initialize import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from .utils import split_tensor_along_last_dim


def _reduce(input_, mpu):
    """All-reduce the the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if mpu.get_tensor_model_parallel_world_size()==1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=mpu.get_tensor_model_parallel_group())

    return input_


def _split(input_, mpu):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = mpu.get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = mpu.get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _gather(input_, mpu):
    """Gather tensors and concatinate along the last dimension."""

    world_size = mpu.get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = mpu.get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=mpu.get_tensor_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, mpu):
        return input_
    
    @staticmethod
    def forward(ctx, input_, mpu):
        ctx.mpu = mpu
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.mpu), None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, mpu):
        return _reduce(input_, mpu)
    
    @staticmethod
    def forward(ctx, input_, mpu):
        return _reduce(input_, mpu)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, mpu):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_, mpu):
        ctx.mpu = mpu
        return _split(input_, mpu)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.mpu), None


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, mpu):
        return _gather(input_, mpu)
    
    @staticmethod
    def forward(ctx, input_, mpu):
        ctx.mpu = mpu
        return _gather(input_, mpu)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.mpu), None


# -----------------
# Helper functions.
# -----------------

def copy_to_tensor_model_parallel_region(input_, mpu):
    return _CopyToModelParallelRegion.apply(input_, mpu)


def reduce_from_tensor_model_parallel_region(input_, mpu):
    return _ReduceFromModelParallelRegion.apply(input_, mpu)


def scatter_to_tensor_model_parallel_region(input_, mpu):
    return _ScatterToModelParallelRegion.apply(input_, mpu)


def gather_from_tensor_model_parallel_region(input_, mpu):
    return _GatherFromModelParallelRegion.apply(input_, mpu)
