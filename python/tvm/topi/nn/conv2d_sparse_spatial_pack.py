import tvm
from tvm import te
from tvm.topi import nn
from tvm.topi.nn.util import get_pad_tuple
from tvm.topi.util import get_const_tuple
from tvm import autotvm
from ..nn.conv2d import conv2d_infer_layout, _get_workload as _get_conv2d_workload
from ..util import get_const_tuple, traverse_inline
from ..x86.conv2d import _get_default_config
from .pad import pad
from ..nn.conv2d import unpack_NCHWc_to_nchw


def csr_spatial_pack_convolution(
        data, indices, indptr, inputs, oshape, kdim, strides, padding, dtype
):
    # pylint: disable=invalid-name
    """The default implementation of csrdc in topi.

    Parameters
    ----------
    data : tvm.te.Tensor
        1-D with shape [nonzeros]

    indices : tvm.te.Tensor
        1-D with shape [nonzeros]

    indptr : tvm.te.Tensor
        1-D with shape [m+1]

    inputs : tvm.te.Tensor
        4-D with shape [N, IC, IH, IW]

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [N, OC, OH, OW]
    """

    KH, KW = kdim
    HSTR, WSTR = strides
    pad_top, pad_left, pad_down, pad_right = padding

    _, ic_chunk, IH, IW, ic_bn = inputs.shape

    def csr_spc_ir(weight_data, indices, indptr, inputs, out):
        """define ir for csrdc"""
        irb = tvm.tir.ir_builder.create()
        weight_data_ptr = irb.buffer_ptr(weight_data)
        indices_ptr = irb.buffer_ptr(indices)
        indptr_ptr = irb.buffer_ptr(indptr)
        inputs_ptr = irb.buffer_ptr(inputs)
        out_ptr = irb.buffer_ptr(out)
        batches, oc_chunk, OH, OW, oc_bn = out.shape


        with irb.for_range(0, batches, name='n') as n:
            with irb.for_range(0, oc_chunk, name='ooc_chunk', for_type="parallel") as ooc_chunk:
                with irb.for_range(0, OH, name='oh') as oh:
                    with irb.for_range(0, OW, name='ow') as ow:
                        with irb.for_range(0, oc_bn, name='oc_block') as oc_block:
                            out_idx = (
                                n*oc_chunk*OH*OW*oc_bn +
                                ooc_chunk*OH*OW*oc_bn +
                                oh*OW*oc_bn +
                                ow*oc_bn +
                                oc_block
                            )
                            out_ptr[out_idx] = 0.

        with irb.for_range(0, batches, name='n') as n:
            with irb.for_range(0, oc_chunk, name='ooc_chunk', for_type="parallel") as ooc_chunk:
                with irb.for_range(0, OH, name='oh') as oh:
                    with irb.for_range(0, OW, name='ow') as ow:
                        with irb.for_range(0, oc_bn, name='oc_block') as oc_block:
                            idx = irb.allocate('int32', (1,), name='idx', scope='local')
                            tmp = irb.allocate('int32', (1,), name='tmp', scope='local')

                            idx[0] = ooc_chunk + oc_block
                            tmp[0] = indptr_ptr[idx[0]+1] - indptr_ptr[idx[0]]

                            with irb.for_range(0, tmp[0], name='j') as j:
                                true_j = irb.allocate('int32', (1,), name='true_j', scope='local')
                                true_j[0] = j + indptr_ptr[idx[0]]
                                off = indices_ptr[true_j[0]]
                                coeff = weight_data_ptr[true_j[0]]
                                kh = (off % (KH * KW)) // KH
                                kw = (off % (KH * KW)) % KW
                                ic = off // (KH * KW)
                                out_idx = (
                                    n*oc_chunk*OH*OW*oc_bn +
                                    ooc_chunk*OH*OW*oc_bn +
                                    oh*OW*oc_bn +
                                    ow*oc_bn +
                                    oc_block
                                )
                                data_idx = (
                                    n*(ic_chunk*IH*IW*ic_bn) +
                                    (ic // ic_bn)*(IH*IW*ic_bn) +
                                    (oh * HSTR + kh)*(IW*ic_bn) +
                                    (ow * WSTR + kw)*ic_bn +
                                    (ic % ic_bn)
                                )
                                out_ptr[out_idx] +=  coeff*inputs_ptr[data_idx]




        print('returning mate')
        return irb.get()

    out = te.extern(oshape, [data, indices, indptr, inputs],
                    lambda ins, outs: csr_spc_ir(ins[0], ins[1], ins[2], ins[3], outs[0]),
                    tag="conv2d_sparse_nchw_spatial_pack", dtype=dtype,
                    name='conv2d_sparse_nchw_spatial_pack')
    print('gretting asshole', type(out))
    return out

def conv2d_sparse_spc_nchw(data, w_data, w_indices, w_indptr,
                           OC, KH, KW, strides, padding, dilation,
                           out_dtype='float32'):
    return conv2d_sparse_sp_NCHW(data, w_data, w_indices, w_indptr,
                                  OC, KH, KW, strides, padding, dilation,
                                  out_dtype)


# @autotvm.register_topi_compute("conv2d.nn")
def conv2d_sparse_sp_NCHW(cfg, data, w_data, w_indices, w_indptr,
                          OC, KH, KW,
                          stride, padding, dilation,
                          out_dtype='float32'):
    n, in_channel, ih, iw = get_const_tuple(data.shape)
    print('sparse sp')
    print(data.shape)

    # Define autotvm tuning space
    kernel_height, kernel_width = KH, KW
    num_filter = OC.__int__()

    print('hello')
    print(type(cfg))
    print(type(OC.__int__()), type(KH), type(KW),)

    is_kernel_1x1 = kernel_height == 1 and kernel_width == 1
    pt, pl, pb, pr = get_pad_tuple(padding, (kernel_height, kernel_width))
    sh, sw = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    oh = (ih - kernel_height + pt + pb) // sh + 1
    ow = (iw - kernel_width + pl + pr) // sw + 1

    cfg.define_split("tile_ic", in_channel, num_outputs=2)
    cfg.define_split("tile_oc", num_filter, num_outputs=2)
    cfg.define_split(
        "tile_ow", ow, num_outputs=2, filter=lambda y: y.size[-1] <= 64, policy="verbose"
    )
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if oh > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])

    # If no config was set, we can fallback to default config.
    if cfg.is_fallback:
        _get_default_config(
            cfg,
            te.placeholder((n, in_channel, ih, iw), dtype=data.dtype),
            te.placeholder(
                (num_filter, in_channel, kernel_height, kernel_width), dtype=out_dtype
            ),
            stride,
            padding,
            out_dtype,
        )
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    # dshape = (n, in_channel // cfg["tile_ic"].size[-1], ih, iw, cfg["tile_ic"].size[-1])



    HSTR, WSTR = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    dilation_h, dilation_w = (
        dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    )

    # reshape data
    dshape = (n, in_channel // cfg["tile_ic"].size[-1], ih, iw, cfg["tile_ic"].size[-1])

    data = te.compute(
        dshape,
        lambda bs, c, h, w, vc: data[bs, c * ic_bn + vc, h, w],
        name="data_vec",
    )


    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    # pad data
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right
    pad_before = (0, 0, pad_top, pad_left, 0)
    pad_after = (0, 0, pad_down, pad_right, 0)

    # DOPAD
    DOPAD = HPAD != 0 or WPAD != 0
    if DOPAD:
        data_pad = pad(data, pad_before, pad_after, name="data_pad")
    else:
        data_pad = data

    kshape = (
        num_filter // cfg["tile_oc"].size[-1],
        in_channel // cfg["tile_ic"].size[-1],
        kernel_height,
        kernel_width,
        cfg["tile_ic"].size[-1],
        cfg["tile_oc"].size[-1],
    )
    # n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
    ic_chunk = in_channel // cfg["tile_ic"].size[-1]

    in_channel = ic_chunk * ic_bn
    target = tvm.target.Target.current(allow_none=False)
    oc_chunk, ic_chunk_group, kernel_height, kernel_width, _, oc_bn = get_const_tuple(kshape)
    num_filter = oc_chunk * oc_bn
    groups = ic_chunk // ic_chunk_group

    # output shape
    out_height = (ih + HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (iw + WPAD - dilated_kernel_w) // WSTR + 1
    # n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
    oshape = (n, oc_chunk, out_height, out_width, oc_bn)

    print('hello friend')
    print(data_pad.shape)
    print(oshape)
    # data = te.compute(
    #     (n, ic_chunk, ih, iw, ic_bn),
    #     lambda bs, c, h, w, vc: data_pad[bs, c * ic_bn + vc, h, w],
    #     name="data_vec",
    # )

    ic = te.reduce_axis((0, in_channel), name="ic")
    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")

    ic = te.reduce_axis((0, in_channel), name="ic")
    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    # conv = te.compute(
    #     oshape,
    #     lambda n, oc_chunk, oh, ow, oc_block: te.sum(
    #         data_pad[
    #             n,
    #             idxdiv(ic, ic_bn),
    #             oh * HSTR + kh * dilation_h,
    #             ow * WSTR + kw * dilation_w,
    #             idxmod(ic, ic_bn),
    #         ].astype(out_dtype)
    #         * 2,
    #         axis=[ic, kh, kw],
    #     ),
    #     name="conv2d_NCHWc",
    #     tag="conv2d_NCHWc",
    # )
    print('smoking weed, breaking hearts')
    print(w_data, w_indices, w_indptr)
    conv = csr_spatial_pack_convolution(
        w_data, w_indices, w_indptr, data_pad, oshape, (KH,KW), (HSTR,WSTR), padding, out_dtype
    )
    oshape = (n, OC, out_height, out_width)

    return unpack_NCHWc_to_nchw(conv, out_dtype)
    # return te.compute(
    #     oshape,
    #     lambda n, oc_chunk, oh, ow, oc_block: te.sum(
    #         data_pad[
    #             n,
    #             idxdiv(ic, ic_bn),
    #             oh * HSTR + kh * dilation_h,
    #             ow * WSTR + kw * dilation_w,
    #             idxmod(ic, ic_bn),
    #         ].astype(out_dtype)
    #         * 2,
    #         axis=[ic, kh, kw],
    #     ),
    #     name="conv2d_NCHWc",
    #     tag="conv2d_NCHWc",
    # )
