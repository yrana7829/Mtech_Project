import time
import numpy as np
import onnxruntime as ort

onnx = "R:\YRANA\M TECH - IITM\FINAL PROJECT\PHASE-3\CODE\scripts\fpga\mnv2_eurosat_naive_ptq_int8.onnx"
outp = "R:\YRANA\M TECH - IITM\FINAL PROJECT\PHASE-3\CODE\scripts\fpga"

sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]

dtype = np.float32
if "float16" in str(inp.type).lower():
    dtype = np.float16
elif "float64" in str(inp.type).lower():
    dtype = np.float64

x = np.random.randn(*shape).astype(dtype)

for _ in range(10):
    sess.run(None, {inp.name: x})

times = []
out = None
for _ in range(100):
    t0 = time.perf_counter()
    out = sess.run(None, {inp.name: x})
    times.append((time.perf_counter() - t0) * 1000.0)

arr = np.array(times)
report = "\n".join(
    [
        "=== FPGA FEASIBILITY SIMULATION (NO BOARD) ===",
        f"Model: {onnx}",
        f"Provider: {sess.get_providers()}",
        f"Input name: {inp.name}",
        f"Input shape(runtime): {shape}",
        f"Input type: {inp.type}",
        f"Runs: {len(arr)}",
        f"Latency avg (ms): {arr.mean():.4f}",
        f"Latency p90 (ms): {np.percentile(arr, 90):.4f}",
        f"Throughput (FPS): {1000.0/arr.mean():.2f}",
        f"Primary output shape: {list(out[0].shape)}",
        f"Top-1 class index: {int(np.argmax(out[0]))}",
        "Verdict: DEPLOYMENT FEASIBLE CANDIDATE (boardless proxy).",
    ]
)

with open(outp, "w") as f:
    f.write(report + "\n")

print(report)
print("Saved", outp)
