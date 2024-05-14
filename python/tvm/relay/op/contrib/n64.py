#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument, use-list-literal
"""DNN64 supported operators for the RSP accelerator.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

import tvm
from tvm import relay
from tvm.target import Target
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

from ..strategy.generic import is_depthwise_conv2d
from .register import register_pattern_table


logger = logging.getLogger("N64")

# tvm._ffi._init_api("relay.ext.n64.transform", __name__)


def is_n64_compiler_enabled() -> bool:
    return "n64" in Target.list_kinds()


def depthwise_conv2d(attrs, args):
    """Check if the external N64 codegen for depthwise convolution should be used.

    Note
    ----
    Relay does not have a depthwise conv2d operator whilst N64 does. We simply
    separate the checks for depthwise for clarity.
    """
    print("\nDepthwise conv!")
    kernel_typ = args[1].checked_type
    # print("hey args:", args)
    print("groups:", attrs.groups)
    if attrs.groups == 1:
        raise ValueError("Error: groups is 1")

    print("kernel shape", kernel_typ.shape)
    print()
    # Only supports 3x3 depthwise
    # if (
    #     kernel_typ.shape[0] not in [3]
    #     or kernel_typ.shape[1] not in [3]
    #     or kernel_typ.shape[0] != kernel_typ.shape[1]
    # ):
    #     print("Depthwise false :(")
    #     return False

    return True


def get_shape(tensor):
    """Get tensor's shape."""
    if isinstance(tensor, relay.expr.Var):
        return tensor.type_annotation.concrete_shape
    if isinstance(tensor, relay.expr.Constant):
        return tensor.data.shape
    if isinstance(tensor, tvm.ir.tensor_type.TensorType):
        return tensor.concrete_shape
    if isinstance(tensor, tvm.ir.container.Array):
        return tensor[-1].shape
    if isinstance(tensor, relay.expr.Call):
        if tensor.op.name == "multiply":
            return tensor.type_args[0].shape
        return tensor.checked_type.shape
    raise TypeError(f"Unsupport data type: {type(tensor)}")


def legalize_depth_conv(attrs, inputs, types):
    """Legalize group conv / conv_transpose calculation.
    Alter weight layout from OIHW to GOIHW / IOHW to GIOHW"""
    if attrs.data_layout != "NHWC":
        raise ValueError(f"Error: data_layout is not NHWC ({attrs.data_layout})")
    if attrs.kernel_layout not in ["HWIO", "HWOI"]:
        raise ValueError(f"Error: kernel_layout is not HWIO or HWOI ({attrs.kernel_layout})")

    groups = attrs.groups
    data, weight = inputs
    H, W, IC, OC = get_shape(weight)
    if groups == 1 or groups != OC:
        print("groups", groups, "oc", OC)
        if "Transpose" not in type(attrs).__name__:
            return relay.nn.conv2d(data, weight, **attrs)
        return relay.nn.conv2d_transpose(data, weight, **attrs)

    assert IC == 1
    assert H == W == 3
    new_attrs = dict(attrs)

    sections = int(OC // 8)  # assume SIMD depth is 8
    split_weights = relay.split(weight, indices_or_sections=sections, axis=-1)
    weight = relay.stack(split_weights, axis=0)
    weight = relay.reshape(weight, (H, W, IC, OC))

    return relay.nn.conv2d(data, weight, **new_attrs)


@tvm.ir.register_op_attr("nn.conv2d", "target.n64")
def _n64_conv2d_wrapper(expr):
    """Check if the external N64 codegen for conv2d should be used."""
    print(f"N64 RSP does support conv2d!.")
    attrs, args = expr.attrs, expr.args
    if attrs.data_layout != "NHWC":  # channels last only
        # TODO force this
        # return False
        ...
    if attrs.out_dtype != "int8" and attrs.out_dtype != "":  # int8 only
        # TODO force this
        ...
    data_typ = args[0].checked_type
    if len(data_typ.shape) != 4 or data_typ.shape[0] != 1 or data_typ.dtype != "float32":
        ...
        # return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 4 or kernel_typ.dtype != "float32":
        ...
        # return False
    is_depthwise = is_depthwise_conv2d(
        data_typ.shape,
        attrs["data_layout"],
        kernel_typ.shape,
        attrs["kernel_layout"],
        attrs["groups"],
    )
    print("hey is_depthwise", is_depthwise)
    if is_depthwise:
        return depthwise_conv2d(attrs, args)
    else:
        return False


@tvm.ir.register_op_attr("qnn.conv2d", "target.n64")
def _n64_qnn_conv2d_wrapper(expr):
    """Check if the external N64 codegen for conv2d should be used."""
    print(f"N64 RSP does support conv2d!.")
    attrs, args = expr.attrs, expr.args
    if attrs.data_layout != "NHWC":  # channels last only
        # TODO force this
        # return False
        ...
    if attrs.out_dtype != "int8" and attrs.out_dtype != "":  # int8 only
        # TODO force this
        ...
    data_typ = args[0].checked_type
    if len(data_typ.shape) != 4 or data_typ.shape[0] != 1 or data_typ.dtype != "float32":
        ...
        # return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 4 or kernel_typ.dtype != "float32":
        ...
        # return False
    is_depthwise = is_depthwise_conv2d(
        data_typ.shape,
        attrs["data_layout"],
        kernel_typ.shape,
        attrs["kernel_layout"],
        attrs["groups"],
    )

    if is_depthwise:
        return depthwise_conv2d(attrs, args)
    else:
        return False


# TODO handle fusion
#
@register_pattern_table("n64")
def pattern_table():
    """Create N64 patterns.

    Returns
    -------
    n64_patterns : List[n64_pattern]
        Created patterns.
    """
    n64_patterns = list()
    return n64_patterns


def partition_for_n64(
    mod: tvm.IRModule,
    params: Optional[Dict[str, tvm.nd.NDArray]] = None,
    target: Optional[tvm.target.Target] = None,
) -> tvm.IRModule:
    """Partition all functions in mod to greedily offload supported operators to TensorRT.

    Parameters
    ----------
    mod : tvm.IRModule
        The module to partition.
    target : tvm.target.Target
        A target of kind "n64" describing additional partitioning and compilation options.
    params : Optional[Dict[str, tvm.nd.NDArray]]
        Constant input parameters.

    Returns
    -------
    partitioned_mod : tvm.IRModule
        The partitioned module.

    """
    assert is_n64_compiler_enabled(), "Can only partition for N64 if it is enabled"
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
    if target is None:
        # Use a default target. The get_n64_target() function will similarly create an
        # equivalent default target when compilation continues after partitioning.
        target = tvm.target.Target("n64")

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.RemoveUnusedFunctions(),
            transform.FoldConstant(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("n64", False),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )
    with target:
        mod = seq(mod)
    return mod
