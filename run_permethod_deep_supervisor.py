import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from run_datasetwise_cemr_framework_experiments import model_specs


DATASETS = ["MIT-BIH", "INCART", "SVDB"]
SEEDS = [303, 1303, 2303, 3303, 4303]


RESULTS = Path("results")
DETAIL = RESULTS / "datasetwise_multimethod_permethod_bioadaptive_detail.csv"
LOG_DIR = RESULTS / "run_logs"
FAILURES = RESULTS / "datasetwise_multimethod_permethod_deep_failures.csv"


def completed_keys():
    if not DETAIL.exists():
        return set()
    df = pd.read_csv(DETAIL)
    if "family" not in df.columns:
        return set()
    deep = df[df["family"].astype(str) == "raw_deep_or_time_series"]
    return set(zip(deep["dataset"].astype(str), deep["method"].astype(str), deep["seed"].astype(int)))


def append_failure(row):
    pd.DataFrame([row]).to_csv(FAILURES, mode="a", header=not FAILURES.exists(), index=False, encoding="utf-8")


def run_child(dataset, method, seed, device_mode, batch_size, timeout_hours):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"{dataset}_{method}_{seed}_{device_mode}_bs{batch_size}_{stamp}".replace("/", "-")
    out_path = LOG_DIR / f"permethod_deep_child_{tag}.out.log"
    err_path = LOG_DIR / f"permethod_deep_child_{tag}.err.log"
    env = os.environ.copy()
    env["CEMR_BATCH_SIZE_OVERRIDE"] = str(int(batch_size))
    # These flags avoid PyTorch Dynamo/compile initialization paths that have
    # caused intermittent Windows APPCRASH failures in long experiment queues.
    env.setdefault("TORCH_DISABLE_DYNAMO", "1")
    env.setdefault("TORCH_COMPILE_DISABLE", "1")
    env.setdefault("CEMR_CUDNN_BENCHMARK", "0")
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    cmd = [
        sys.executable,
        "run_datasetwise_multimethod_permethod_bioadaptive_framework.py",
        "--families",
        "deep",
        "--datasets",
        dataset,
        "--methods",
        method,
        "--seeds",
        str(int(seed)),
    ]
    if device_mode == "cpu":
        cmd.append("--cpu")
    with out_path.open("w", encoding="utf-8") as out_f, err_path.open("w", encoding="utf-8") as err_f:
        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                stdout=out_f,
                stderr=err_f,
                cwd=Path.cwd(),
                env=env,
                timeout=float(timeout_hours) * 3600.0,
            )
            code = int(proc.returncode)
        except subprocess.TimeoutExpired:
            code = 124
        elapsed = time.time() - start
    return code, elapsed, out_path, err_path


def run(args):
    methods = args.methods or [s.name for s in model_specs()]
    datasets = args.datasets or DATASETS
    seeds = args.seeds or SEEDS
    attempts = []
    if not args.cpu_only:
        for bs in args.gpu_batch_sizes:
            attempts.append(("cuda", int(bs)))
    for bs in args.cpu_batch_sizes:
        attempts.append(("cpu", int(bs)))

    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                key = (dataset, method, int(seed))
                if key in completed_keys() and not args.force:
                    print(f"Skip completed {dataset} {method} seed={seed}", flush=True)
                    continue
                print(f"Supervisor running {dataset} {method} seed={seed}", flush=True)
                success = False
                for device_mode, batch_size in attempts:
                    if key in completed_keys() and not args.force:
                        success = True
                        break
                    print(f"  attempt device={device_mode} batch_size={batch_size}", flush=True)
                    code, elapsed, out_path, err_path = run_child(
                        dataset,
                        method,
                        int(seed),
                        device_mode,
                        batch_size,
                        args.timeout_hours,
                    )
                    if key in completed_keys():
                        print(
                            f"  completed code={code} elapsed={elapsed:.1f}s out={out_path.name}",
                            flush=True,
                        )
                        success = True
                        break
                    print(
                        f"  failed/no-row code={code} elapsed={elapsed:.1f}s out={out_path.name} err={err_path.name}",
                        flush=True,
                    )
                    append_failure(
                        {
                            "dataset": dataset,
                            "method": method,
                            "seed": int(seed),
                            "device_mode": device_mode,
                            "batch_size": int(batch_size),
                            "return_code": int(code),
                            "elapsed_sec": float(elapsed),
                            "out_log": str(out_path),
                            "err_log": str(err_path),
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
                if not success:
                    print(f"Supervisor could not complete {dataset} {method} seed={seed}", flush=True)
                    if args.stop_on_failure:
                        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="Crash-resilient supervisor for deep/time-series per-method CEMR runs.")
    parser.add_argument("--datasets", nargs="*", default=None, choices=DATASETS)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--gpu-batch-sizes", nargs="*", type=int, default=[128, 64, 32])
    parser.add_argument("--cpu-batch-sizes", nargs="*", type=int, default=[64])
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=4.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
