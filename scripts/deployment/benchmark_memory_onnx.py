import argparse
import os
import time

import numpy as np
import onnxruntime as ort
import psutil
import platform
import subprocess


def main(args):

    print("\n==========================================")
    print(" Raspberry Pi Memory Benchmark")
    print("==========================================")

    print(f"Host              : {platform.node()}")
    print(f"Architecture      : {platform.machine()}")
    print(f"Python            : {platform.python_version()}")
    print(f"ONNX Runtime      : {ort.__version__}")

    try:
        model = subprocess.check_output(
            ["cat", "/proc/device-tree/model"], text=True
        ).strip("\x00")
        print(f"Hardware          : {model}")
    except Exception:
        pass

    process = psutil.Process(os.getpid())

    session = ort.InferenceSession(
        args.model,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

    print("\nONNX Runtime:", ort.__version__)
    print("Providers:", session.get_providers())

    # ----------------------------------------------------
    # Warmup
    # ----------------------------------------------------

    print(f"\nWarmup ({args.warmup} runs)...")

    for _ in range(args.warmup):
        session.run(None, {input_name: dummy})

    # ----------------------------------------------------
    # Benchmark
    # ----------------------------------------------------

    print(f"\nRunning {args.runs} benchmark inferences...\n")

    inference_times = []
    memory_usage = []

    start_total = time.perf_counter()

    for _ in range(args.runs):

        start = time.perf_counter()

        session.run(None, {input_name: dummy})

        end = time.perf_counter()

        inference_times.append((end - start) * 1000)

        rss = process.memory_info().rss / (1024 * 1024)

        memory_usage.append(rss)

    total_time = time.perf_counter() - start_total

    cpu_after = psutil.cpu_percent(interval=1)

    # ----------------------------------------------------
    # Statistics
    # ----------------------------------------------------

    inference_times = np.array(inference_times)

    memory_usage = np.array(memory_usage)

    model_size = os.path.getsize(args.model) / (1024 * 1024)

    throughput = args.runs / total_time

    print("\n==========================================")
    print(" RESULTS")
    print("==========================================")

    print(f"Model                : {args.model}")
    print(f"Model Size           : {model_size:.2f} MB")

    print()

    print(f"Mean Latency         : {np.mean(inference_times):.3f} ms")
    print(f"Median Latency       : {np.median(inference_times):.3f} ms")
    print(f"Std Latency          : {np.std(inference_times):.3f} ms")
    print(f"P95 Latency          : {np.percentile(inference_times,95):.3f} ms")

    print()

    print(f"Average RSS Memory   : {np.mean(memory_usage):.2f} MB")
    print(f"Peak RSS Memory      : {np.max(memory_usage):.2f} MB")
    print(f"Minimum RSS Memory   : {np.min(memory_usage):.2f} MB")

    print()

    print(f"CPU Utilization      : {cpu_after:.1f}%")

    print(f"Throughput           : {throughput:.2f} images/sec")

    print(f"Total Benchmark Time : {total_time:.2f} s")

    print("==========================================\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    main(args)
