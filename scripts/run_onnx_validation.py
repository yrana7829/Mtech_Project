import argparse
import os
import sys

import numpy as np
import torch
import onnxruntime as ort

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.models.model_loader import get_model
from src.dataset.dataloader import get_dataset


def validate(args):

    device = torch.device("cpu")

    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_dataset(args.dataset)

    print("Loading PyTorch model...")
    model = get_model(args.model, num_classes=args.num_classes)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    model.load_state_dict(checkpoint)
    model.eval()

    print("Loading ONNX model...")
    session = ort.InferenceSession(args.onnx_model)

    print("Execution Providers:")
    print(session.get_providers())

    # Get one batch
    images, labels = next(iter(test_loader))

    # Use first image
    image = images[0:1]

    print("Input shape:", image.shape)

    # ---------------------------
    # PyTorch inference
    # ---------------------------

    with torch.no_grad():

        pytorch_output = model(image)

    pytorch_output_np = pytorch_output.numpy()

    # ---------------------------
    # ONNX inference
    # ---------------------------

    ort_inputs = {session.get_inputs()[0].name: image.numpy().astype(np.float32)}

    onnx_output = session.run(None, ort_inputs)[0]

    # ---------------------------
    # Compare outputs
    # ---------------------------

    max_diff = np.max(np.abs(pytorch_output_np - onnx_output))

    mean_diff = np.mean(np.abs(pytorch_output_np - onnx_output))

    pytorch_pred = np.argmax(pytorch_output_np)

    onnx_pred = np.argmax(onnx_output)

    print("\n===== RESULTS =====")

    print(f"PyTorch Prediction: {pytorch_pred}")

    print(f"ONNX Prediction: {onnx_pred}")

    print(f"Prediction Match: " f"{pytorch_pred == onnx_pred}")

    print(f"Max Difference: {max_diff:.10f}")

    print(f"Mean Difference: {mean_diff:.10f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)

    parser.add_argument("--dataset", required=True)

    parser.add_argument("--num_classes", type=int, required=True)

    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--onnx_model", required=True)

    args = parser.parse_args()

    validate(args)
