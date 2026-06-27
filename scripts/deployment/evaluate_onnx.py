import argparse
import os
import sys
import time

import numpy as np
import torch
import onnxruntime as ort

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.models.model_loader import get_model
from src.dataset.dataloader import get_dataset


def evaluate_onnx(args):

    device = torch.device("cpu")

    print("Loading dataset...")
    _, _, test_loader = get_dataset(args.dataset)

    print("Loading PyTorch model...")

    model = get_model(args.model, num_classes=args.num_classes)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    model.load_state_dict(checkpoint)
    model.eval()

    print("Loading ONNX model...")

    session = ort.InferenceSession(args.onnx_model)

    print("Execution Providers:")
    print(session.get_providers())

    input_name = session.get_inputs()[0].name

    # ----------------------------------
    # Accuracy variables
    # ----------------------------------

    total = 0

    pytorch_correct = 0
    onnx_correct = 0

    prediction_matches = 0

    max_diff_list = []
    mean_diff_list = []

    # ----------------------------------
    # Latency variables
    # ----------------------------------

    pytorch_times = []
    onnx_times = []

    print("\nRunning evaluation...\n")

    with torch.no_grad():

        for images, labels in test_loader:

            batch_size = images.size(0)

            # -------------------------
            # PyTorch
            # -------------------------

            start = time.perf_counter()

            pytorch_output = model(images)

            pytorch_time = time.perf_counter() - start

            pytorch_times.append(pytorch_time / batch_size)

            pytorch_output_np = pytorch_output.numpy()

            pytorch_preds = np.argmax(pytorch_output_np, axis=1)

            # -------------------------
            # ONNX
            # -------------------------

            ort_inputs = {input_name: images.numpy().astype(np.float32)}

            start = time.perf_counter()

            onnx_output = session.run(None, ort_inputs)[0]

            onnx_time = time.perf_counter() - start

            onnx_times.append(onnx_time / batch_size)

            onnx_preds = np.argmax(onnx_output, axis=1)

            # -------------------------
            # Accuracy
            # -------------------------

            labels_np = labels.numpy()

            pytorch_correct += np.sum(pytorch_preds == labels_np)

            onnx_correct += np.sum(onnx_preds == labels_np)

            prediction_matches += np.sum(pytorch_preds == onnx_preds)

            total += batch_size

            # -------------------------
            # Output differences
            # -------------------------

            max_diff = np.max(np.abs(pytorch_output_np - onnx_output))

            mean_diff = np.mean(np.abs(pytorch_output_np - onnx_output))

            max_diff_list.append(max_diff)
            mean_diff_list.append(mean_diff)

    # ----------------------------------
    # Final Metrics
    # ----------------------------------

    pytorch_acc = 100.0 * pytorch_correct / total

    onnx_acc = 100.0 * onnx_correct / total

    agreement = 100.0 * prediction_matches / total

    avg_pytorch_latency = np.mean(pytorch_times) * 1000

    avg_onnx_latency = np.mean(onnx_times) * 1000

    model_size_mb = os.path.getsize(args.onnx_model) / (1024 * 1024)

    print("\n========== RESULTS ==========\n")

    print(f"PyTorch Accuracy : " f"{pytorch_acc:.2f}%")

    print(f"ONNX Accuracy    : " f"{onnx_acc:.2f}%")

    print(f"Prediction Agreement : " f"{agreement:.2f}%")

    print()

    print(f"Avg PyTorch Latency : " f"{avg_pytorch_latency:.3f} ms/image")

    print(f"Avg ONNX Latency    : " f"{avg_onnx_latency:.3f} ms/image")

    print()

    print(f"Model Size : " f"{model_size_mb:.2f} MB")

    print()

    print(f"Avg Max Difference : " f"{np.mean(max_diff_list):.6f}")

    print(f"Avg Mean Difference : " f"{np.mean(mean_diff_list):.6f}")

    print("\n=============================\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)

    parser.add_argument("--dataset", required=True)

    parser.add_argument("--num_classes", type=int, required=True)

    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--onnx_model", required=True)

    args = parser.parse_args()

    evaluate_onnx(args)
