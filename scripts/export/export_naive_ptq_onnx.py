import argparse
import os
import sys
import random

import numpy as np
import torch
import onnx

from torch.utils.data import Subset, DataLoader
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.insert(0, PROJECT_ROOT)

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate

# ============================================================
# FIXED EXPERIMENT CONFIGURATION
# ============================================================

CALIB_SIZE = 1000
CALIB_SEED = 42

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

torch.backends.quantized.engine = "fbgemm"


# ============================================================
# BUILD FIXED CALIBRATION LOADER
# ============================================================


def build_calibration_loader(train_loader):

    train_dataset = train_loader.dataset

    generator = torch.Generator()
    generator.manual_seed(CALIB_SEED)

    indices = torch.randperm(len(train_dataset), generator=generator)[:CALIB_SIZE]

    calib_dataset = Subset(train_dataset, indices)

    calib_loader = DataLoader(
        calib_dataset, batch_size=train_loader.batch_size, shuffle=False, num_workers=0
    )

    return calib_loader


# ============================================================
# FX PTQ
# ============================================================


def build_naive_ptq_model(model, calibration_loader):

    device = torch.device("cpu")

    model.eval()
    model.to(device)

    qconfig = get_default_qconfig("fbgemm")

    qconfig_dict = {"": qconfig}

    example_inputs = next(iter(calibration_loader))[0][:1].to(device)

    print("\nPreparing FX PTQ model...")

    prepared_model = prepare_fx(model, qconfig_dict, example_inputs)

    print("Running calibration...")

    with torch.no_grad():

        for images, _ in calibration_loader:

            images = images.to(device)

            prepared_model(images)

    print("Converting to real INT8 model...")

    quantized_model = convert_fx(prepared_model)

    quantized_model.eval()

    return quantized_model


# ============================================================
# INSPECT QUANTIZED PYTORCH MODEL
# ============================================================


def inspect_quantized_modules(model):

    quantized_count = 0

    for _, module in model.named_modules():

        module_type = str(type(module)).lower()

        if "quantized" in module_type:
            quantized_count += 1

    return quantized_count


# ============================================================
# EXPORT
# ============================================================


def export_to_onnx(model, output_path):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dummy_input = torch.randn(1, 3, 224, 224)

    print("\nExporting quantized model to ONNX...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )

    print(f"ONNX model saved to:\n{output_path}")


# ============================================================
# ONNX STRUCTURAL VALIDATION
# ============================================================


def validate_onnx_structure(onnx_path):

    print("\nChecking ONNX model validity...")

    model = onnx.load(onnx_path)

    onnx.checker.check_model(model)

    print("ONNX model is valid.")

    op_counts = {}

    for node in model.graph.node:

        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    print("\n===== ONNX GRAPH SUMMARY =====")

    print("Nodes:", len(model.graph.node))

    print("Initializers:", len(model.graph.initializer))

    print("\nOperator counts:")

    for op, count in sorted(op_counts.items()):

        print(f"{op:25s}: {count}")

    return op_counts


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)

    parser.add_argument("--model", required=True)

    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    device = torch.device("cpu")

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print("Loading dataset...")

    train_loader, val_loader, test_loader = get_dataset(args.dataset)

    calibration_loader = build_calibration_loader(train_loader)

    print(f"Calibration samples: " f"{len(calibration_loader.dataset)}")

    print(f"Test samples: " f"{len(test_loader.dataset)}")

    # --------------------------------------------------------
    # FP32 MODEL
    # --------------------------------------------------------

    print("\nLoading FP32 model...")

    model = get_model(args.model, num_classes=args.num_classes)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    model.load_state_dict(checkpoint)

    model.eval()

    # --------------------------------------------------------
    # CREATE ACTUAL PTQ MODEL
    # --------------------------------------------------------

    quant_model = build_naive_ptq_model(model, calibration_loader)

    quantized_module_count = inspect_quantized_modules(quant_model)

    print("\nQuantized modules detected:", quantized_module_count)

    # --------------------------------------------------------
    # IMPORTANT:
    # EVALUATE THE PYTORCH PTQ MODEL ON TEST SET
    # --------------------------------------------------------

    print("\nEvaluating PyTorch FX PTQ " "on TEST SET...")

    ptq_accuracy = evaluate(quant_model, test_loader, device)

    print(f"\nPyTorch FX PTQ Test Accuracy: " f"{ptq_accuracy * 100:.2f}%")

    # --------------------------------------------------------
    # EXPORT THE EXACT MODEL JUST EVALUATED
    # --------------------------------------------------------

    export_to_onnx(quant_model, args.output)

    # --------------------------------------------------------
    # STRUCTURAL VALIDATION
    # --------------------------------------------------------

    op_counts = validate_onnx_structure(args.output)

    # --------------------------------------------------------
    # FILE SIZE
    # --------------------------------------------------------

    model_size_mb = os.path.getsize(args.output) / (1024**2)

    print(f"\nONNX file size: " f"{model_size_mb:.2f} MB")

    # --------------------------------------------------------
    # FINAL EXPORT RECORD
    # --------------------------------------------------------

    print("\n" "========================================")

    print("EXPORT COMPLETE")

    print("========================================")

    print(f"Method               : Naive FX PTQ INT8")

    print(f"PyTorch PTQ Accuracy : " f"{ptq_accuracy * 100:.2f}%")

    print(f"Quantized Modules    : " f"{quantized_module_count}")

    print(f"ONNX File            : " f"{args.output}")

    print(f"ONNX Size            : " f"{model_size_mb:.2f} MB")

    print("ONNX Valid           : YES")

    print("========================================")


if __name__ == "__main__":
    main()
