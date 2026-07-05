import time
import subprocess
import numpy as np
import onnxruntime as ort

FP32_MODEL = "models/eurosat_mobilenetv2_fp32.onnx"
INT8_MODEL = "models/mnv2_fx_ptq.onnx"

WARMUP = 50
TIMED_RUNS = 500
REPEATS = 5


def get_temperature():
    try:
        output = subprocess.check_output(
            ["vcgencmd", "measure_temp"], text=True
        ).strip()

        return float(output.replace("temp=", "").replace("'C", ""))

    except Exception:
        return float("nan")


def create_session(model_path):

    options = ort.SessionOptions()

    return ort.InferenceSession(
        model_path, sess_options=options, providers=["CPUExecutionProvider"]
    )


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


def print_result(run_number, model_name, result):

    print(
        f"Run {run_number} | {model_name:10s} | "
        f"Mean {result['mean']:.3f} ms | "
        f"Median {result['median']:.3f} ms | "
        f"Std {result['std']:.3f} ms | "
        f"P95 {result['p95']:.3f} ms | "
        f"Temp {result['temp_before']:.1f}"
        f"->{result['temp_after']:.1f} C"
    )


def main():

    print("\nCreating sessions...")

    fp32_session = create_session(FP32_MODEL)
    int8_session = create_session(INT8_MODEL)

    print("FP32 providers:", fp32_session.get_providers())
    print("INT8 providers:", int8_session.get_providers())

    rng = np.random.default_rng(42)

    input_tensor = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)

    fp32_results = []
    int8_results = []

    print("\nStarting controlled benchmark...\n")

    for run in range(1, REPEATS + 1):

        # Alternate execution order to reduce order bias
        if run % 2 == 1:
            order = [
                ("FP32", fp32_session),
                ("INT8", int8_session),
            ]
        else:
            order = [
                ("INT8", int8_session),
                ("FP32", fp32_session),
            ]

        for name, session in order:

            result = benchmark(session, input_tensor)

            if name == "FP32":
                fp32_results.append(result)
            else:
                int8_results.append(result)

            print_result(run, name, result)

    print("\n==========================================")
    print(" FINAL SUMMARY")
    print("==========================================")

    fp32_means = np.array([result["mean"] for result in fp32_results])

    int8_means = np.array([result["mean"] for result in int8_results])

    fp32_mean = np.mean(fp32_means)
    int8_mean = np.mean(int8_means)

    print(f"FP32 mean latency : " f"{fp32_mean:.3f} ± {np.std(fp32_means):.3f} ms")

    print(f"INT8 mean latency : " f"{int8_mean:.3f} ± {np.std(int8_means):.3f} ms")

    print(f"INT8 / FP32 ratio : " f"{int8_mean / fp32_mean:.3f}x")

    print(f"FP32 / INT8 speedup: " f"{fp32_mean / int8_mean:.3f}x")

    difference = 100.0 * (int8_mean - fp32_mean) / fp32_mean

    print(f"INT8 latency change: " f"{difference:+.2f}%")

    print("==========================================\n")


if __name__ == "__main__":
    main()
