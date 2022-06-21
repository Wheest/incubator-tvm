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
def kernel_atax(N: int, M: int, out_dtype: str = "float32"):
    A = te.placeholder((M, N), name="A")
    x = te.placeholder((N,), name="x")

    # for (i = 0; i < _PB_M; i++)
    #   {
    #     tmp[i] = SCALAR_VAL(0.0);
    #     for (j = 0; j < _PB_N; j++)
    #       tmp[i] = tmp[i] + A[i][j] * x[j];
    j = te.reduce_axis((0, N), "j")
    tmp = te.compute(
        (M,),
        lambda i: te.sum(
            A[i][j] * x[j],
            axis=j,
        ),
    )

    # for (j = 0; j < _PB_N; j++)
    #     y[j] = y[j] + A[i][j] * tmp[i];
    i = te.reduce_axis((0, M), "i")
    y = te.compute(
        (N,),
        lambda j: te.sum(
            A[i][j] * tmp[i],
            axis=i,
        ),
    )

    return [A, x, y]


def cpu_default_schedule(out):
    s = te.create_schedule(out.op)
    return s


def np_atax(N, M, A, x, y):
    tmp = np.zeros(M)
    for i in range(N):
        y[i] = 0
    for i in range(M):
        for j in range(N):
            tmp[i] = tmp[i] + A[i][j] * x[j]
        for j in range(N):
            y[j] = y[j] + A[i][j] * tmp[i]

    return y


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
        N = M = args.size
    else:
        N, M = args.N, args.M
        for v in [N, M]:
            assert type(v) == int, "if N, or K is set, all must be set"

    # Get the TVM IR represnetation of the program
    logger.info("Creating kernel")
    func = kernel_atax
    func_args = [N, M, args.dtype]
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
            func, func_args, "atax", target, log_file.name, args.ntrials
        )
        log_file.close()  # delete log file when we are finished
    else:
        # compile with default schedule
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.build(s, tvm_args, target=target, name="atax")
    logger.info("Compiled program")

    # Generate input data, and target output data
    A_np = np.random.uniform(size=(M, N)).astype(np.float32)
    x_np = np.random.uniform(size=(N)).astype(np.float32)
    y_np = np.random.uniform(size=(N)).astype(np.float32)
    target = np_atax(N, M, A_np, x_np, y_np)

    # transfer to TVM objects
    A_tvm = tvm.nd.array(A_np.astype(args.dtype), device=dev)
    x_tvm = tvm.nd.array(x_np.astype(args.dtype), device=dev)
    # y_tvm = tvm.nd.array(y_np.astype(args.dtype), device=dev)

    # allocate empty array for output
    out_tvm = tvm.nd.empty((N,), device=dev)

    # run the compiled program
    lib_args = [A_tvm, x_tvm, out_tvm]

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
    parser = argparse.ArgumentParser(description="Polybench-TVM atax")
    parser.add_argument("--size", type=int, help="Size of matrices", default=32)
    parser.add_argument("--N", type=int, help="Size N of matrices", default=None)
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
    print(f"atax: {med_time}")
