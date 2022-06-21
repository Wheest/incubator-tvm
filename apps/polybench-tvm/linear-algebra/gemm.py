import os
import argparse
import logging
from typing import Optional, List
import tempfile
import numpy as np

from scipy.linalg.blas import dgemm
import tvm
from tvm import te, auto_scheduler

import sys

module_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
if module_path not in sys.path:
    sys.path.append(module_path)
from utils import auto_schedule_func


@auto_scheduler.register_workload
def kernel_gemm(N: int, K: int, M: int, out_dtype: str = "float32"):
    A = te.placeholder((N, K), name="A")
    B = te.placeholder((K, M), name="B")
    C = te.placeholder((N, M), name="C")
    alpha = te.placeholder((1,), name="alpha")
    beta = te.placeholder((1,), name="beta")
    k = te.reduce_axis((0, K), "k")

    # Reductions are only allowed at the top level of compute.
    # Hence why we have this placeholder
    tmp = te.compute((N, M), lambda n, m: beta[0] * C[n, m])

    out_C = te.compute(
        (N, M),
        lambda n, m: te.sum(
            alpha[
                0,
            ]
            * A[n, k]
            * B[k, m],
            axis=k,
        ),
        name="matmul",
    )

    out_C = te.compute((N, M), lambda n, m: out_C[n, m] + tmp[n, m], name="out")

    return [A, B, C, alpha, beta, out_C]


def cpu_default_schedule(out):
    s = te.create_schedule(out.op)
    return s


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
        N = K = M = args.size
    else:
        N, K, M = args.N, args.K, args.M
        for v in [N, K, M]:
            assert type(v) == int, "if N, M, or K is set, all must be set"

    # Get the TVM IR represnetation of the program
    logger.info("Creating kernel")
    func = kernel_gemm
    func_args = [N, K, M, args.dtype]
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
            func, func_args, "gemm", target, log_file.name, args.ntrials
        )
        log_file.close()  # delete log file when we are finished
    else:
        # compile with default schedule
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.build(s, tvm_args, target=target, name="gemm")
    logger.info("Compiled program")

    # Generate input data, and target output data
    a_np = np.random.uniform(size=(N, K)).astype(np.float32)
    b_np = np.random.uniform(size=(K, M)).astype(np.float32)
    alpha_np = np.random.uniform(size=(1,)).astype(np.float32)
    c_np = np.random.uniform(size=(N, M)).astype(np.float32)
    beta_np = np.random.uniform(size=(1,)).astype(np.float32)
    target = dgemm(alpha_np, a_np, b_np, c=c_np, beta=beta_np)

    # transfer to TVM objects
    a_tvm = tvm.nd.array(a_np.astype(args.dtype), device=dev)
    b_tvm = tvm.nd.array(b_np.astype(args.dtype), device=dev)
    c_tvm = tvm.nd.array(c_np.astype(args.dtype), device=dev)
    alpha_tvm = tvm.nd.array(alpha_np.astype(args.dtype), device=dev)
    beta_tvm = tvm.nd.array(beta_np.astype(args.dtype), device=dev)

    # allocate empty array for output
    out_tvm = tvm.nd.empty((N, M), device=dev)

    # run the compiled program
    lib_args = [a_tvm, b_tvm, c_tvm, alpha_tvm, beta_tvm, out_tvm]

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
    parser = argparse.ArgumentParser(description="Polybench-TVM gemm")
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
    print(f"GEMM: {med_time}")
