import os
import argparse
import logging
from typing import Optional, List
import tempfile
import numpy as np

import tvm
from tvm import te, auto_scheduler

import sys

module_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
if module_path not in sys.path:
    sys.path.append(module_path)
from utils import auto_schedule_func


@auto_scheduler.register_workload
def kernel_gemver(N: int, out_dtype: str = "float32"):
    # D := alpha*A*B*C + beta*D

    # initialize placeholder arrays
    A = te.placeholder((N, N), name="A")
    u1 = te.placeholder((N,), name="u1")
    u2 = te.placeholder((N,), name="u2")
    v1 = te.placeholder((N,), name="v1")
    v2 = te.placeholder((N,), name="v2")
    w = te.placeholder((N,), name="w")
    x = te.placeholder((N,), name="x")
    y = te.placeholder((N,), name="y")
    z = te.placeholder((N,), name="z")

    alpha = te.placeholder((1,), name="alpha")
    beta = te.placeholder((1,), name="beta")

    # for (i = 0; i < _PB_N; i++)
    #   for (j = 0; j < _PB_N; j++)
    #     A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
    A = te.compute((N, N), lambda i, j: A[i][j] + (u1[i] * v1[j]) + (u2[i] * v2[j]))

    # for (i = 0; i < _PB_N; i++)
    #   for (j = 0; j < _PB_N; j++)
    #     x[i] = x[i] + beta * A[j][i] * y[j];
    # ??? this should mean that we only need j=J?
    j = te.reduce_axis((0, N), "j")
    x = te.compute(
        N,
        te.sum(
            x[i] + beta[0] * A[j][i] * y[j]
            axis=j,
        ),
    )


    return [A, B, C, D, alpha, beta, out_D]


def cpu_default_schedule(out):
    s = te.create_schedule(out.op)
    return s


def np_gemver(A, u1, u2, v1, v2, w, x, y, z, alpha, beta):
    A = (
        A
        + u1[:, np.newaxis] * v1[np.newaxis, :]
        + u2[:, np.newaxis] * v2[np.newaxis, :]
    )

    x = x + beta * A.dot(y)
    x = x + z
    w = w + (alpha * np.dot(A, x))
    return w


def main(args):
    logger = logging.getLogger("tvm-polybench")
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    )

    if args.verbose:
        logger.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if args.N is None:
        NI = NJ = NK = NL = args.size
    else:
        NI, NJ, NK, NL = args.NI, args.NJ, args.NK, args.NL
        for v in [NI, NJ, NK, NL]:
            assert type(v) == int, "if any NI, NJ, NK, or NL is set, all must be set"

    # Get the TVM IR represnetation of the program
    logger.info("Creating kernel")
    func = kernel_2mm
    func_args = [NI, NJ, NK, NL, args.dtype]
    tvm_args = func(*func_args)
    logger.info("Created kernel")

    # get the schedule, target, and device
    if args.backend == "opencl":
        s = gemm_gpu_schedule(tvm_args[-1])
        target = tvm.target.Target("opencl -device=mali", host="llvm -mcpu=core-avx2")
        dev = tvm.cl()
    elif args.backend == "cpu":
        s = cpu_default_schedule(tvm_args[-1])
        target = tvm.target.Target("llvm -mcpu=core-avx2")
        dev = tvm.cpu()
    else:
        raise ValueError("Unexpected backend:", args.backend)
    logger.info("Created schedule")

    if args.auto_schedule:
        # auto-schedule if desired
        logging.info("Will auto-schedule this function")
        log_file = tempfile.NamedTemporaryFile(suffix=".json")
        lib = auto_schedule_func(
            func, func_args, "2mm", target, log_file.name, args.ntrials
        )
        log_file.close()  # delete log file when we are finished
    else:
        # compile with default schedule
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.build(s, tvm_args, target=target, name="2mm")
    logger.info("Compiled program")

    # Generate input data, and target output data
    alpha_np = np.random.uniform(size=(1,)).astype(np.float32)

    beta_np = np.random.uniform(size=(1,)).astype(np.float32)

    A_np = np.random.uniform(size=(N, N)).astype(np.float32)
    u1_np = np.random.uniform(size=(N)).astype(np.float32)
    u2_np = np.random.uniform(size=(N)).astype(np.float32)
    v1_np = np.random.uniform(size=(N)).astype(np.float32)
    v2_np = np.random.uniform(size=(N)).astype(np.float32)
    w_np = np.random.uniform(size=(N)).astype(np.float32)
    x_np = np.random.uniform(size=(N)).astype(np.float32)
    y_np = np.random.uniform(size=(N)).astype(np.float32)
    z_np = np.random.uniform(size=(N)).astype(np.float32)

    target = np_gemver(
        A_np, u1_np, u2_np, v1_np, v2_np, w_np, x_np, y_np, z_np, alpha_np, beta_np
    )

    # transfer to TVM objects
    A_tvm = tvm.nd.array(A_np.astype(args.dtype), device=dev)
    u1_tvm = tvm.nd.array(u1_np.astype(args.dtype), device=dev)
    u2_tvm = tvm.nd.array(u2_np.astype(args.dtype), device=dev)
    v1_tvm = tvm.nd.array(v1_np.astype(args.dtype), device=dev)
    v2_tvm = tvm.nd.array(v2_np.astype(args.dtype), device=dev)
    w_tvm = tvm.nd.array(w_np.astype(args.dtype), device=dev)
    x_tvm = tvm.nd.array(x_np.astype(args.dtype), device=dev)
    y_tvm = tvm.nd.array(y_np.astype(args.dtype), device=dev)
    z_tvm = tvm.nd.array(z_np.astype(args.dtype), device=dev)
    alpha_tvm = tvm.nd.array(alpha_np.astype(args.dtype), device=dev)
    beta_tvm = tvm.nd.array(beta_np.astype(args.dtype), device=dev)

    # allocate empty array for output
    out_tvm = tvm.nd.empty((N), device=dev)

    # run the compiled program
    lib_args = [
        A_tvm,
        u1_tvm,
        u2_tvm,
        v1_tvm,
        v2_tvm,
        w_tvm,
        x_tvm,
        y_tvm,
        z_tvm,
        alpha_tvm,
        beta_tvm,
    ]

    lib(*lib_args)

    # assert correctness
    np.testing.assert_allclose(target, out_tvm.asnumpy(), rtol=1e-3)

    # Evaluate execution time
    evaluator = lib.time_evaluator(lib.entry_name, dev, min_repeat_ms=500)
    res = evaluator(*lib_args).results
    med_time = np.median(res) * 1000
    std_time = np.std(res) * 1000
    mean_time = np.mean(res) * 1000
    logger.info("Execution time of this operator: %.3f ms" % (med_time))

    return med_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polybench-TVM gemmver")
    parser.add_argument("--size", type=int, help="Size of matrices", default=32)
    parser.add_argument("--N", type=int, help="Size N of matrices", default=None)
    parser.add_argument("--K", type=int, help="Size M of matrices", default=None)
    parser.add_argument("--M", type=int, help="Size M of matrices", default=None)

    parser.add_argument(
        "--backend",
        default="cpu",
        choices=["cpu", "opencl"],
        help="Backend to compile to",
    )
    parser.add_argument(
        "--dtype", type=str, help="Data type to compute with", default="float32"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug messages to console",
    )
    parser.add_argument(
        "--auto_schedule",
        action="store_true",
        help="Run the auto-scheduler for this computation",
    )
    parser.add_argument(
        "--ntrials",
        type=int,
        help="Number of candidate programs for the auto-scheduler to evaluate",
        default=200,
    )
    args = parser.parse_args()

    med_time = main(args)
    print(f"gemver: {med_time}")
