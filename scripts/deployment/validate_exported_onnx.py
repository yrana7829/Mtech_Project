import argparse
import os
import sys

import numpy as np
import onnxruntime as ort

# ============================================================
# PROJECT PATH
# ============================================================

# add project root to python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.insert(0, PROJECT_ROOT)

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate

# ============================================================
# ONNX EVALUATION
# ============================================================


def evaluate_onnx(session, test_loader):

    input_info = session.get_inputs()[0]

    input_name = input_info.name

    total = 0
    correct = 0

    print("\nRunning ONNX evaluation...")

    for images, labels in test_loader:

        # Fixed-batch ONNX export:
        # process one image at a time.

        for i in range(images.size(0)):

            image = images[i : i + 1].numpy().astype(np.float32)

            label = int(labels[i].item())

            output = session.run(None, {input_name: image})[0]

            prediction = int(np.argmax(output, axis=1)[0])

            if prediction == label:
                correct += 1

            total += 1

    accuracy = 100.0 * correct / total

    return (accuracy, correct, total)


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)

    parser.add_argument("--onnx_model", required=True)

    parser.add_argument(
        "--reference_accuracy",
        type=float,
        required=True,
        help=("PyTorch quantized model accuracy " "measured immediately before export"),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print("Loading dataset...")

    _, _, test_loader = get_dataset(args.dataset)

    print("Test samples:", len(test_loader.dataset))

    # --------------------------------------------------------
    # ONNX RUNTIME
    # --------------------------------------------------------

    print("\nLoading ONNX model...")

    session = ort.InferenceSession(args.onnx_model, providers=["CPUExecutionProvider"])

    print("\nExecution Providers:")

    print(session.get_providers())

    input_info = session.get_inputs()[0]

    output_info = session.get_outputs()[0]

    print("\nInput:")

    print(input_info.name, input_info.shape, input_info.type)

    print("\nOutput:")

    print(output_info.name, output_info.shape, output_info.type)

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    onnx_accuracy, correct, total = evaluate_onnx(session, test_loader)

    accuracy_difference = onnx_accuracy - args.reference_accuracy

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print("\n" "========================================")

    print("EXPORT VALIDATION RESULT")

    print("========================================")

    print(f"Reference PyTorch PTQ Accuracy : " f"{args.reference_accuracy:.2f}%")

    print(f"Exported ONNX Accuracy         : " f"{onnx_accuracy:.2f}%")

    print(f"Accuracy Difference            : " f"{accuracy_difference:+.2f} pp")

    print(f"Correct / Total                : " f"{correct} / {total}")

    print("========================================")

    if abs(accuracy_difference) < 1e-9:

        print("\nRESULT: Exact accuracy preservation.")

    else:

        print("\nRESULT: Export changed model accuracy.")


if __name__ == "__main__":
    main()
