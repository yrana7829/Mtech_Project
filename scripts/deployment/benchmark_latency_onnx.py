import time
import subprocess
import numpy as np
import onnxruntime as ort
import platform

# ==========================================================
# MODELS
# ==========================================================

FP32_MODEL = "models/eurosat_mobilenetv2_fp32.onnx"
NAIVE_MODEL = "models/mnv2_eurosat_naive_ptq_int8.onnx"
PTQPP_MODEL = "models/mnv2_eurosat_ptqpp_v3_int8.onnx"

# ==========================================================
# BENCHMARK SETTINGS
# ==========================================================

WARMUP = 50
TIMED_RUNS = 500
REPEATS = 5


# ==========================================================
# TEMPERATURE
# ==========================================================


def get_temperature():
    try:
        output = subprocess.check_output(
            ["vcgencmd", "measure_temp"],
            text=True,
        ).strip()

        return float(output.replace("temp=", "").replace("'C", ""))

    except Exception:
        return float("nan")


# ==========================================================
# SYSTEM INFORMATION
# ==========================================================


def print_system_info():

    print("\n==========================================")
    print(" Raspberry Pi Deployment Environment")
    print("==========================================")

    print(f"Host              : {platform.node()}")
    print(f"Architecture      : {platform.machine()}")
    print(f"Python            : {platform.python_version()}")
    print(f"ONNX Runtime      : {ort.__version__}")

    try:
        model = subprocess.check_output(
            ["cat", "/proc/device-tree/model"],
            text=True,
        ).strip("\x00")

        print(f"Hardware          : {model}")

    except Exception:
        pass

    print("==========================================")


# ==========================================================
# CREATE SESSION
# ==========================================================


def create_session(model_path):

    options = ort.SessionOptions()

    return ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


# ==========================================================
# BENCHMARK
# ==========================================================


def benchmark(session, input_tensor):

    input_name = session.get_inputs()[0].name

    # Warm-up
    for _ in range(WARMUP):
        session.run(None, {input_name: input_tensor})

    times = []

    temp_before = get_temperature()

    for _ in range(TIMED_RUNS):

        start = time.perf_counter_ns()

        session.run(None, {input_name: input_tensor})

        end = time.perf_counter_ns()

        times.append((end - start) / 1_000_000)

    temp_after = get_temperature()

    times = np.array(times)

    return {
        "mean": np.mean(times),
        "median": np.median(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "p95": np.percentile(times, 95),
        "temp_before": temp_before,
        "temp_after": temp_after,
    }


# ==========================================================
# PRINT RESULT
# ==========================================================


def print_result(run_number, model_name, result):

    print(
        f"Run {run_number} | "
        f"{model_name:15s} | "
        f"Mean {result['mean']:.3f} ms | "
        f"Median {result['median']:.3f} ms | "
        f"Std {result['std']:.3f} ms | "
        f"P95 {result['p95']:.3f} ms | "
        f"Temp {result['temp_before']:.1f}"
        f"->{result['temp_after']:.1f} C"
    )


# ==========================================================
# MAIN
# ==========================================================


def main():

    print_system_info()

    print("\nCreating ONNX Runtime sessions...\n")

    fp32_session = create_session(FP32_MODEL)
    naive_session = create_session(NAIVE_MODEL)
    ptqpp_session = create_session(PTQPP_MODEL)

    print("FP32 Provider     :", fp32_session.get_providers())
    print("Naive PTQ Provider:", naive_session.get_providers())
    print("PTQ++ Provider    :", ptqpp_session.get_providers())

    rng = np.random.default_rng(42)

    input_tensor = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)

    fp32_results = []
    naive_results = []
    ptqpp_results = []

    print("\nStarting controlled benchmark...\n")

    for run in range(1, REPEATS + 1):

        # Alternate execution order
        if run % 2 == 1:

            order = [
                ("FP32", fp32_session),
                ("Naive PTQ", naive_session),
                ("Proposed PTQ++", ptqpp_session),
            ]

        else:

            order = [
                ("Proposed PTQ++", ptqpp_session),
                ("Naive PTQ", naive_session),
                ("FP32", fp32_session),
            ]

        for name, session in order:

            result = benchmark(session, input_tensor)

            if name == "FP32":
                fp32_results.append(result)

            elif name == "Naive PTQ":
                naive_results.append(result)

            else:
                ptqpp_results.append(result)

            print_result(run, name, result)

    # ======================================================
    # SUMMARY
    # ======================================================

    fp32_means = np.array([x["mean"] for x in fp32_results])
    naive_means = np.array([x["mean"] for x in naive_results])
    ptqpp_means = np.array([x["mean"] for x in ptqpp_results])

    fp32_mean = np.mean(fp32_means)
    naive_mean = np.mean(naive_means)
    ptqpp_mean = np.mean(ptqpp_means)

    print("\n==========================================")
    print(" FINAL BENCHMARK SUMMARY")
    print("==========================================")

    print(f"FP32 Mean Latency          : {fp32_mean:.3f} ± {np.std(fp32_means):.3f} ms")
    print(
        f"Naive PTQ Mean Latency     : {naive_mean:.3f} ± {np.std(naive_means):.3f} ms"
    )
    print(
        f"Proposed PTQ++ Latency     : {ptqpp_mean:.3f} ± {np.std(ptqpp_means):.3f} ms"
    )

    print()

    print(f"Naive PTQ Speedup          : {fp32_mean / naive_mean:.3f}x")
    print(f"Proposed PTQ++ Speedup     : {fp32_mean / ptqpp_mean:.3f}x")

    print()

    naive_change = 100.0 * (naive_mean - fp32_mean) / fp32_mean
    ptqpp_change = 100.0 * (ptqpp_mean - fp32_mean) / fp32_mean

    print(f"Naive PTQ Latency Change   : {naive_change:+.2f}%")
    print(f"Proposed PTQ++ Change      : {ptqpp_change:+.2f}%")

    print("==========================================\n")


if __name__ == "__main__":
    main()
