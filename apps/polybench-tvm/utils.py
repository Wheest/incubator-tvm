#!/usr/bin/env python
import tvm
import os
from tvm import auto_scheduler
from typing import Optional, List


def auto_schedule_func(
    func,
    func_args: List,
    func_name: str,
    target,
    log_file: os.PathLike,
    ntrials: int = 200,
    tuning_time: Optional[int] = None,
):
    task = tvm.auto_scheduler.SearchTask(func=func, args=func_args, target=target)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=ntrials,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    # Run auto-tuning (search)
    task.tune(tune_option)

    # Apply the best schedule
    sch, args = task.apply_best(log_file)

    # Build the new function
    lib = tvm.build(sch, args, target, name=func_name)

    return lib
