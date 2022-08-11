#!/usr/bin/env python
import tvm
from tvm import te
from tvm.topi import nn
from ..utils import get_const_tuple
from functools import partial, reduce

def sparse_conv2d_direct_compute_nchw(
    data,
    w_data,
    w_indices,
    w_indptr,
    kernel_size,
    padding,
    strides,
    dilation,
    mode="CPU",
):
    batches, IC, IH, IW = get_const_tuple(data.shape)

    if isinstance(kernel_size, int):
        KH = KW = kernel_size
    else:
        KH, KW = kernel_size

    if len(w_data.shape) == 2:
        _, bs_r = get_const_tuple(w_data.shape)
    elif len(w_data.shape) == 3:
        _, bs_r, bs_c = get_const_tuple(w_data.shape)

    (num_blocks_plus_1,) = get_const_tuple(w_indptr.shape)
    num_blocks = num_blocks_plus_1 - 1

    OC = num_blocks * bs_r

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1

    if len(padding) == 4:
        pad_top, pad_left, pad_down, pad_right = padding
    else:
        pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
            padding, (dilated_kernel_h, dilated_kernel_w)
        )
    if isinstance(strides, (tuple, list)):
        if len(strides) == 2:
            HSTR, WSTR = strides
        elif len(strides) == 4:
            HSTR, _, WSTR, _ = strides
        else:
            raise ValueError(f"Unexpected strides size: `{strides}`")
    else:
        HSTR, WSTR = strides

    OH = (IH + pad_top + pad_down - dilated_kernel_h) // HSTR + 1
    OW = (IW + pad_left + pad_right - dilated_kernel_w) // WSTR + 1

    if pad_top or pad_left:
        data_pad = nn.pad(
            data,
            [0, 0, pad_top, pad_left],
            [0, 0, pad_down, pad_right],
            name="data_pad",
        )
    else:
        data_pad = data

    oshape = (batches, OC, OH, OW)
    # out = csr_direct_convolution(
    #     w_data, w_indices, w_indptr, data_pad, oshape, (KH, KW), (HSTR, WSTR), mode=mode
    # )
    @partial(
        te.compute,
        (batches, OC, OH, OW),
        name="CC",
        tag="sparse_direct_conv2d",
        attrs={"layout": "NCHW"},
    )
    def direct_conv2d(n, oc, oh, ow):
        row_start, row_end = w_indptr[oc], w_indptr[oc + 1]
        elem_idx = te.reduce_axis((0, row_end - row_start), name="elem_idx")
        elem = row_start + elem_idx
        # coeff = w_data[elem]

        # calculate indice values for input data
        offset = w_indices[elem]
        kh = (offset % (KH * KW)) // KH
        kw = (offset % (KH * KW)) % KW

        ic = offset // (KH * KW)
        ih = oh * HSTR + kh
        iw = ow * WSTR + kw

        return te.sum(
            data_pad[n, ic, ih, iw] * w_data[elem, 0, 0],
            axis=elem_idx,
        )

    return direct_conv2d


def schedule_sparse_conv2d_direct_nchw_cpu(s, conv_out, data):
    print("default CPU sparse conv2d direct schedule")
    # no schedule
    return s
