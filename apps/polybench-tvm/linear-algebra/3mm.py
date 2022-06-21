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
def kernel_3mm(NI: int, NJ: int, NK: int, NL: int, NM: int, out_dtype: str = "float32"):

    A = te.placeholder((NI, NK), name="A")
    B = te.placeholder((NK, NJ), name="B")
    C = te.placeholder((NJ, NL), name="C")
    D = te.placeholder((NI, NL), name="D")

    # compute E := A*B
    k = te.reduce_axis((0, NK), "k")

    E = te.compute(
        (NI, NJ),
        lambda i, j: te.sum(
            A[i, k] * B[k, j],
            axis=k,
        ),
    )

    # compute F := C*D
    k = te.reduce_axis((0, NM), "k")

    F = te.compute(
        (NJ, NL),
        lambda i, j: te.sum(
            C[i, k] * D[k, j],
            axis=k,
        ),
    )

    # compute G := E*F
    k = te.reduce_axis((0, NJ), "k")

    G = te.compute(
        (NI, NL),
        lambda i, j: te.sum(
            E[i, k] * F[k, j],
            axis=k,
        ),
    )

    return [A, B, C, D, G]


def cpu_default_schedule(out):
    s = te.create_schedule(out.op)
    return s


def np_3mm(A, B, C, D, E, F, G):
    # run the 3mm kernel in numpy
    # E := A*B
    # F := C*D
    # G := E*F
    # return G
    E = np.dot(A, B)
    F = np.dot(C, D)
    G = np.dot(E, F)
    return G


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
        NI = NJ = NK = NL = NM = args.size
    else:
        NI, NJ, NK, NL = args.NI, args.NJ, args.NK, args.NL, args.NM
        for v in [NI, NJ, NK, NL]:
            assert type(v) == int, "if any NI, NJ, NK, or NL is set, all must be set"

    # Get the TVM IR represnetation of the program
    logger.info("Creating kernel")
    func = kernel_3mm
    func_args = [NI, NJ, NK, NL, NM, args.dtype]
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
            func, func_args, "3mm", target, log_file.name, args.ntrials
        )
        log_file.close()  # delete log file when we are finished
    else:
        # compile with default schedule
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.build(s, tvm_args, target=target, name="3mm")
    logger.info("Compiled program")

    # Generate input data, and target output data
    a_np = np.random.uniform(size=(NI, NK)).astype(np.float32)
    b_np = np.random.uniform(size=(NK, NJ)).astype(np.float32)
    c_np = np.random.uniform(size=(NJ, NM)).astype(np.float32)
    d_np = np.random.uniform(size=(NM, NL)).astype(np.float32)
    e_np = np.zeros(shape=(NI, NJ)).astype(np.float32)
    f_np = np.zeros(shape=(NJ, NL)).astype(np.float32)
    g_np = np.zeros(shape=(NI, NL)).astype(np.float32)
    target = np_3mm(a_np, b_np, c_np, d_np, e_np, f_np, g_np)

    # transfer to TVM objects
    a_tvm = tvm.nd.array(a_np.astype(args.dtype), device=dev)
    b_tvm = tvm.nd.array(b_np.astype(args.dtype), device=dev)
    c_tvm = tvm.nd.array(c_np.astype(args.dtype), device=dev)
    d_tvm = tvm.nd.array(d_np.astype(args.dtype), device=dev)

    # allocate empty array for output
    out_tvm = tvm.nd.empty((NI, NL), device=dev)

    # run the compiled program
    lib_args = [a_tvm, b_tvm, c_tvm, d_tvm, out_tvm]

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
    parser = argparse.ArgumentParser(description="Polybench-TVM 3mm")
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
    print(f"3mm: {med_time}")
