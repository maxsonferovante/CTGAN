"""Benchmark CTGAN and TVAE on the bundled Adult dataset using the GPU.

Fits both synthesizers on ``examples/csv/adult.csv``, generates more than
1,000,000 synthetic rows with each one and reports timing and memory usage.

Run it directly with:

    uv run test.py
"""

import ctypes
import json
import os
import platform
import time

import pandas as pd
import torch

from ctgan import CTGAN, TVAE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "examples", "csv", "adult.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "benchmark_output")
N_SAMPLES = 1_100_000
EPOCHS = 50
BATCH_SIZE = 500
DISCRETE_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "income",
]


def resolve_device():
    """Return the device CTGAN and TVAE pick when ``enable_gpu=True``."""
    if torch.cuda.is_available():
        return "cuda"

    if (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def process_rss_mb():
    """Best-effort resident memory of this process, in MB."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        pass

    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1024 if platform.system() != "Darwin" else rss / 1e6
    except Exception:
        pass

    if platform.system() == "Windows":
        try:

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            )
            return counters.WorkingSetSize / 1e6
        except Exception:
            pass

    return float("nan")


def gpu_memory_mb():
    """Return CUDA memory stats in MB, or ``None`` when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return None

    return {
        "allocated": torch.cuda.memory_allocated() / 1e6,
        "reserved": torch.cuda.memory_reserved() / 1e6,
        "peak_allocated": torch.cuda.max_memory_allocated() / 1e6,
        "peak_reserved": torch.cuda.max_memory_reserved() / 1e6,
    }


def run(name, model, data, n_samples, output_dir):
    """Fit ``model``, sample rows, save them to CSV and report timing and memory."""
    print(f"\n--- {name} ---")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    rss_before = process_rss_mb()

    fit_start = time.perf_counter()
    model.fit(data, DISCRETE_COLUMNS)
    fit_time = time.perf_counter() - fit_start

    sample_start = time.perf_counter()
    synthetic = model.sample(n_samples)
    sample_time = time.perf_counter() - sample_start

    csv_path = os.path.join(output_dir, f"synthetic_{name.lower()}.csv")
    save_start = time.perf_counter()
    synthetic.to_csv(csv_path, index=False)
    save_time = time.perf_counter() - save_start

    rss_after = process_rss_mb()
    gpu = gpu_memory_mb()
    rows = len(synthetic)
    throughput = rows / sample_time if sample_time else float("inf")
    csv_size_mb = os.path.getsize(csv_path) / 1e6

    print(f"device               : {model._device}")
    print(f"rows generated       : {rows:,}")
    print(f"fit time             : {fit_time:,.2f} s")
    print(f"sample time          : {sample_time:,.2f} s")
    print(f"sample throughput    : {throughput:,.0f} rows/s")
    print(f"total time           : {fit_time + sample_time:,.2f} s")
    print(f"CPU RSS before/after : {rss_before:,.1f} / {rss_after:,.1f} MB")
    if gpu:
        print(
            f'GPU allocated        : {gpu["allocated"]:,.1f} MB '
            f'(reserved {gpu["reserved"]:,.1f} MB)'
        )
        print(
            f'GPU peak allocated   : {gpu["peak_allocated"]:,.1f} MB '
            f'(reserved {gpu["peak_reserved"]:,.1f} MB)'
        )
    print(
        f"csv saved            : {csv_path} ({csv_size_mb:,.1f} MB, {save_time:,.2f} s)"
    )

    return {
        "model": name,
        "device": str(model._device),
        "rows": rows,
        "fit_time_s": round(fit_time, 3),
        "sample_time_s": round(sample_time, 3),
        "save_time_s": round(save_time, 3),
        "total_time_s": round(fit_time + sample_time + save_time, 3),
        "rows_per_second": round(throughput, 1),
        "cpu_rss_before_mb": round(rss_before, 1),
        "cpu_rss_after_mb": round(rss_after, 1),
        "gpu_peak_allocated_mb": round(gpu["peak_allocated"], 1) if gpu else None,
        "gpu_peak_reserved_mb": round(gpu["peak_reserved"], 1) if gpu else None,
        "csv_path": csv_path,
        "csv_size_mb": round(csv_size_mb, 1),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = resolve_device()
    print(f"torch    : {torch.__version__}")
    print(f"device   : {device}")
    gpu_name = None
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"gpu      : {gpu_name}")
    elif device == "cpu":
        print("warning  : no GPU detected, running on CPU")

    data = pd.read_csv(DATA_PATH)
    print(f"dataset  : {DATA_PATH}")
    print(f"rows/cols: {data.shape[0]:,} / {data.shape[1]}")

    results = [
        run(
            "CTGAN",
            CTGAN(epochs=EPOCHS, batch_size=BATCH_SIZE, enable_gpu=True),
            data,
            N_SAMPLES,
            OUTPUT_DIR,
        )
    ]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results.append(
        run(
            "TVAE",
            TVAE(epochs=EPOCHS, batch_size=BATCH_SIZE, enable_gpu=True),
            data,
            N_SAMPLES,
            OUTPUT_DIR,
        )
    )

    print("\n=== summary ===")
    print(
        f'{"model":<8}{"rows":>12}{"fit (s)":>12}{"sample (s)":>14}'
        f'{"total (s)":>12}{"gpu peak MB":>14}'
    )
    for result in results:
        gpu_peak = result["gpu_peak_allocated_mb"]
        gpu_peak = f"{gpu_peak:,.1f}" if gpu_peak is not None else "n/a"
        print(
            f'{result["model"]:<8}{result["rows"]:>12,}'
            f'{result["fit_time_s"]:>12,.2f}{result["sample_time_s"]:>14,.2f}'
            f'{result["total_time_s"]:>12,.2f}{gpu_peak:>14}'
        )

    report = {
        "torch_version": torch.__version__,
        "device": device,
        "gpu": gpu_name,
        "dataset": DATA_PATH,
        "dataset_rows": int(data.shape[0]),
        "dataset_columns": int(data.shape[1]),
        "discrete_columns": DISCRETE_COLUMNS,
        "requested_samples": N_SAMPLES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "results": results,
    }
    json_path = os.path.join(OUTPUT_DIR, "benchmark_report.json")
    with open(json_path, "w") as report_file:
        json.dump(report, report_file, indent=2)

    print(f"\njson report : {json_path}")
    for result in results:
        print(f'csv output  : {result["csv_path"]}')


if __name__ == "__main__":
    main()
