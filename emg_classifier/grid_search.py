#!/usr/bin/env python3
"""
grid_search.py – standalone multi-GPU grid search runner.

Called from explore_emg.ipynb via subprocess to avoid Jupyter + multiprocessing
incompatibilities (spawn re-imports __main__ = the Jupyter kernel, which
crashes; fork is safe here because CUDA is not initialised before the fork).

Tasks are read from a JSON file whose path is passed as --tasks.
Each task is a list matching the run_task() argument tuple in emg_worker.py.

Usage:
    python grid_search.py --tasks /tmp/tasks.json --n-workers 7
"""
import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Ensure emg_worker is importable when called from any working directory.
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from emg_worker import run_task  # noqa: E402


def _fix_task(t: list) -> tuple:
    """JSON deserialises tuples as lists; restore the kernels tuple."""
    t = list(t)
    t[2] = tuple(t[2])   # kernels: list → tuple
    return tuple(t)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="JSON file with task list")
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    with open(args.tasks) as f:
        tasks = [_fix_task(t) for t in json.load(f)]

    n = len(tasks)
    if n == 0:
        print("No tasks to run.")
        return

    if args.n_workers == 1:
        for i, t in enumerate(tasks, 1):
            try:
                r = run_task(t)
                print(f"[{i}/{n}] done  subject={r['subject']}  acc={r['final_acc']:.4f}",
                      flush=True)
            except Exception as e:
                print(f"[{i}/{n}] ERROR  task={t[:5]}  {e}", flush=True)
    else:
        # fork is safe here: no CUDA and no Jupyter state in this process.
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx) as pool:
            futures = {pool.submit(run_task, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                t = futures[fut]
                try:
                    r = fut.result()
                    print(f"[{i}/{n}] done  subject={r['subject']}  acc={r['final_acc']:.4f}",
                          flush=True)
                except Exception as e:
                    print(f"[{i}/{n}] ERROR  task={t[:5]}  {e}", flush=True)


if __name__ == "__main__":
    main()
