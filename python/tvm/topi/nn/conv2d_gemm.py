import tvm
from tvm import te
from tvm.topi import nn
from tvm.topi.nn.util import get_pad_tuple
from tvm.topi.util import get_const_tuple
from tvm import autotvm
from ..util import traverse_inline, get_const_tuple


def conv2d_gemm_nchw(data, weights, strides, padding, dilation,
                     layout):
    return conv2d_gemm_NCHW(data, weights, strides, padding, dilation,
                            layout)

@autotvm.register_topi_compute("conv2d_gemm.nn")
def conv2d_gemm_NCHW(cfg, data, weights, strides, padding, dilation,
                     layout, out_dtype='float32'):
    """Compute conv2d by transforming the input,
    executing GEMM and not transforming the output back yet"""
    batches, IC, IH, IW = get_const_tuple(data.shape)
    OC, IC, KH, KW = get_const_tuple(weights.shape)

    K = KH * KW

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = \
        get_pad_tuple(padding, (dilated_kernel_h, dilated_kernel_w))
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)

    OH = (IH + pad_top + pad_down - dilated_kernel_h) // HSTR + 1
    OW = (IW + pad_left + pad_right - dilated_kernel_w) // WSTR + 1

    N = OC
    K = KH * KW * IC
    M = OH * OW


    # --- Weight reshape
    A = tvm.topi.nn.flatten(weights)

    if pad_top or pad_left:
        data_pad = nn.pad(data, [0, 0, pad_top, pad_left], [0, 0, pad_down, pad_right],
                          name="data_pad")
    else:
        data_pad = data

    # --- Im2col (transposed)

    B_shape = (batches, M, K)
    idxmod = tvm.tir.indexmod
    idxdiv = tvm.tir.indexdiv

    B = te.compute(B_shape, lambda n, m, k:
                   data_pad[n, (k // (KH*KW)) % IC,
                            (k // KH) % KW + ((m // OW) * HSTR),
                            (k % KW) + ((m % OW) * WSTR)],
                       name='data_im2col')


    # --- GEMM: A*B'
    k = te.reduce_axis((0, K), 'k')
    # C = te.compute(
    #            (batches, N, M),
    #            lambda b, n, m: te.sum(A[n, k] * B[b, k, m], axis=k),
    #            name='C')

    oshape = (batches, OC, OH, OW)
    # C = te.compute(
    #     oshape,
    #     lambda b, c, h, w: C[b, c, h*OW + w]
    # )

    # C = te.compute(
    #     oshape,
    #     lambda b, c, h, w: te.sum(A[c, k] * B[b, k, h*OW + w], axis=k),
    #     name='C')
    #
    # C = te.compute(
    #     oshape,
    #     lambda b, c, h, w: te.sum(A[c, k] * B[b, h*OW + w, k], axis=k),
    #     name='C')
    #
    #    print('hey partner', N,K,M)
    cfg.define_split("tile_y", 32 if isinstance(M, tvm.tir.Var) else M, num_outputs=2)
    cfg.define_split("tile_x", 32 if isinstance(N, tvm.tir.Var) else N, num_outputs=2)
    cfg.define_split("tile_k", 32 if isinstance(K, tvm.tir.Var) else K, num_outputs=2)
    vec = cfg["tile_k"].size[-1]
    k = te.reduce_axis((0, K // vec), "k")

    CC = te.compute(
        (batches, N, M, vec),
        lambda n, z, y, x: te.sum(
            # A[z, 0].astype(out_dtype) * B[n, 0, 0].astype(out_dtype),
            A[z, k * vec + x].astype(out_dtype) * B[n, y, k * vec + x].astype(out_dtype),
            axis=k,
        ),
    )
    kk = te.reduce_axis((0, vec), "kk")
    # C = te.compute((batches, N, M), lambda n, y, x: te.sum(CC[n, y, x, kk], axis=kk), tag="dense_nopack")
    C = te.compute((batches, OC, OH, OW),
                   lambda b, c, h, w: te.sum(CC[b, 0, 0, kk], axis=kk),
                   tag="dense_nopack")
    return C


@autotvm.register_topi_schedule("conv2d_NCHW.x86")
def schedule_gemm_conv2d_nchw(cfg, outs):
    """Create schedule for tensors"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "dense_nopack" in op.tag:
            _schedule_gemm_pack_template(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_gemm_pack_template(cfg, s, C):
    _, y, OH, OW = s[C].op.axis
    x = s[C].fuse(OH, OW)
    (kk,) = s[C].op.reduce_axis
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(yo, xo, yi, xi)
    xyo = s[C].fuse(yo, xo)
    s[C].parallel(xyo)
    s[C].unroll(kk)

    (CC,) = s[C].op.input_tensors
    s[CC].compute_at(s[C], xyo)
    _, z, y, x = s[CC].op.axis
    (k,) = s[CC].op.reduce_axis
    yz = s[CC].fuse(z, y)
    s[CC].reorder(k, yz, x)
    s[CC].unroll(yz)
    s[CC].vectorize(x)
    return s
