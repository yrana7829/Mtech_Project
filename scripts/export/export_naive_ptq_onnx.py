import argparse
import os
import sys
import random

import numpy as np
import torch
import onnx
import onnxruntime as ort

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
GLOBAL_SEED = 42

torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

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
# BUILD NAIVE FX PTQ MODEL
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

    quantized_modules = []

    for name, module in model.named_modules():

        module_type = str(type(module)).lower()

        if "quantized" in module_type:

            quantized_modules.append((name, str(type(module))))

    print("\n========================================")

    print("QUANTIZED PYTORCH MODEL INSPECTION")

    print("========================================")

    print(f"Total Quantized Modules: " f"{len(quantized_modules)}")

    print("========================================")

    return len(quantized_modules)


# ============================================================
# EXPORT TO ONNX
# ============================================================


def export_to_onnx(model, output_path):

    output_directory = os.path.dirname(output_path)

    if output_directory:

        os.makedirs(output_directory, exist_ok=True)

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

    print("\nONNX export completed.")

    print(f"Saved to: {output_path}")


# ============================================================
# ONNX STRUCTURAL VALIDATION
# ============================================================


def validate_onnx_structure(onnx_path):

    print("\nChecking ONNX model validity...")

    onnx_model = onnx.load(onnx_path)

    onnx.checker.check_model(onnx_model)

    print("ONNX model is valid.")

    op_counts = {}

    for node in onnx_model.graph.node:

        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    print("\n========================================")

    print("ONNX GRAPH SUMMARY")

    print("========================================")

    print("Nodes:", len(onnx_model.graph.node))

    print("Initializers:", len(onnx_model.graph.initializer))

    print("\nOperator Counts:")

    for op_name, count in sorted(op_counts.items()):

        print(f"{op_name:25s}: {count}")

    print("========================================")

    return op_counts


# ============================================================
# PAIRED EXPORT FIDELITY COMPARISON
# ============================================================


def compare_pytorch_ptq_and_onnx(quant_model, onnx_path, test_loader):

    quant_model.eval()

    print("\nLoading exported ONNX model " "for paired comparison...")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    print("Execution Providers:", session.get_providers())

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    input_name = input_info.name

    print("\nONNX Input:")

    print(input_info.name, input_info.shape, input_info.type)

    print("\nONNX Output:")

    print(output_info.name, output_info.shape, output_info.type)

    total = 0

    pytorch_correct = 0
    onnx_correct = 0

    prediction_matches = 0

    pytorch_correct_onnx_wrong = 0
    pytorch_wrong_onnx_correct = 0

    max_differences = []
    mean_differences = []

    print("\nComparing exact PyTorch PTQ model " "with exported ONNX model...")

    with torch.no_grad():

        for images, labels in test_loader:

            batch_size = images.size(0)

            for i in range(batch_size):

                # --------------------------------------------
                # ONE IMAGE
                # --------------------------------------------

                image = images[i : i + 1]

                label = int(labels[i].item())

                # --------------------------------------------
                # EXACT PYTORCH PTQ MODEL
                # --------------------------------------------

                pytorch_output = quant_model(image)

                pytorch_output_np = pytorch_output.detach().cpu().numpy()

                pytorch_pred = int(np.argmax(pytorch_output_np, axis=1)[0])

                # --------------------------------------------
                # EXPORTED ONNX MODEL
                # --------------------------------------------

                onnx_input = image.cpu().numpy().astype(np.float32)

                onnx_output = session.run(None, {input_name: onnx_input})[0]

                onnx_pred = int(np.argmax(onnx_output, axis=1)[0])

                # --------------------------------------------
                # ACCURACY
                # --------------------------------------------

                pytorch_is_correct = pytorch_pred == label

                onnx_is_correct = onnx_pred == label

                if pytorch_is_correct:

                    pytorch_correct += 1

                if onnx_is_correct:

                    onnx_correct += 1

                # --------------------------------------------
                # PREDICTION AGREEMENT
                # --------------------------------------------

                if pytorch_pred == onnx_pred:

                    prediction_matches += 1

                # --------------------------------------------
                # DIRECTION OF EXPORT-INDUCED CHANGES
                # --------------------------------------------

                if pytorch_is_correct and not onnx_is_correct:

                    pytorch_correct_onnx_wrong += 1

                if not pytorch_is_correct and onnx_is_correct:

                    pytorch_wrong_onnx_correct += 1

                # --------------------------------------------
                # NUMERICAL OUTPUT DIFFERENCE
                # --------------------------------------------

                absolute_difference = np.abs(pytorch_output_np - onnx_output)

                max_differences.append(float(np.max(absolute_difference)))

                mean_differences.append(float(np.mean(absolute_difference)))

                total += 1

    # ========================================================
    # FINAL METRICS
    # ========================================================

    pytorch_accuracy = 100.0 * pytorch_correct / total

    onnx_accuracy = 100.0 * onnx_correct / total

    accuracy_difference = onnx_accuracy - pytorch_accuracy

    agreement = 100.0 * prediction_matches / total

    changed_predictions = total - prediction_matches

    print("\n========================================")

    print("PAIRED EXPORT FIDELITY RESULT")

    print("========================================")

    print(f"Total Test Images                  : " f"{total}")

    print()

    print(f"PyTorch PTQ Accuracy               : " f"{pytorch_accuracy:.2f}%")

    print(f"Exported ONNX Accuracy             : " f"{onnx_accuracy:.2f}%")

    print(f"Accuracy Difference                : " f"{accuracy_difference:+.2f} pp")

    print()

    print(f"Prediction Agreement               : " f"{agreement:.2f}%")

    print(f"Predictions Changed by Export      : " f"{changed_predictions} / {total}")

    print()

    print(f"PyTorch Correct -> ONNX Wrong      : " f"{pytorch_correct_onnx_wrong}")

    print(f"PyTorch Wrong -> ONNX Correct      : " f"{pytorch_wrong_onnx_correct}")

    print()

    print(f"Average Mean Output Difference     : " f"{np.mean(mean_differences):.6f}")

    print(f"Average Max Output Difference      : " f"{np.mean(max_differences):.6f}")

    print(f"Worst Max Output Difference        : " f"{np.max(max_differences):.6f}")

    print("========================================")

    return {
        "total": total,
        "pytorch_accuracy": pytorch_accuracy,
        "onnx_accuracy": onnx_accuracy,
        "accuracy_difference": accuracy_difference,
        "prediction_agreement": agreement,
        "changed_predictions": changed_predictions,
        "pytorch_correct_onnx_wrong": pytorch_correct_onnx_wrong,
        "pytorch_wrong_onnx_correct": pytorch_wrong_onnx_correct,
        "average_mean_difference": float(np.mean(mean_differences)),
        "average_max_difference": float(np.mean(max_differences)),
        "worst_max_difference": float(np.max(max_differences)),
    }


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

    # ========================================================
    # DATASET
    # ========================================================

    print("Loading dataset...")

    train_loader, val_loader, test_loader = get_dataset(args.dataset)

    calibration_loader = build_calibration_loader(train_loader)

    print(f"Calibration samples: " f"{len(calibration_loader.dataset)}")

    print(f"Test samples: " f"{len(test_loader.dataset)}")

    # ========================================================
    # LOAD FP32 MODEL
    # ========================================================

    print("\nLoading FP32 model...")

    model = get_model(args.model, num_classes=args.num_classes)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    model.load_state_dict(checkpoint)

    model.eval()

    # ========================================================
    # BUILD ACTUAL NAIVE FX PTQ MODEL
    # ========================================================

    quant_model = build_naive_ptq_model(model, calibration_loader)

    # ========================================================
    # INSPECT QUANTIZED MODULES
    # ========================================================

    quantized_module_count = inspect_quantized_modules(quant_model)

    # ========================================================
    # EVALUATE PYTORCH PTQ MODEL
    # ========================================================

    print("\nEvaluating PyTorch FX PTQ " "on TEST SET...")

    ptq_accuracy = evaluate(quant_model, test_loader, device)

    print(f"\nPyTorch FX PTQ Test Accuracy: " f"{ptq_accuracy * 100:.2f}%")

    # ========================================================
    # EXPORT EXACT MODEL JUST EVALUATED
    # ========================================================

    export_to_onnx(quant_model, args.output)

    # ========================================================
    # CHECK ONNX STRUCTURE
    # ========================================================

    op_counts = validate_onnx_structure(args.output)

    # ========================================================
    # FILE SIZE
    # ========================================================

    model_size_mb = os.path.getsize(args.output) / (1024**2)

    print(f"\nONNX File Size: " f"{model_size_mb:.2f} MB")

    # ========================================================
    # CRITICAL PAIRED COMPARISON
    #
    # Uses:
    #   - exact quant_model still in memory
    #   - exact ONNX file exported from that object
    #   - exact same test_loader
    # ========================================================

    fidelity_results = compare_pytorch_ptq_and_onnx(
        quant_model=quant_model, onnx_path=args.output, test_loader=test_loader
    )

    # ========================================================
    # FINAL PIPELINE SUMMARY
    # ========================================================

    print("\n\n" "========================================")

    print("FINAL NAIVE PTQ EXPORT SUMMARY")

    print("========================================")

    print("Method:")

    print("Naive FX PTQ INT8")

    print()

    print(f"Calibration Samples       : " f"{len(calibration_loader.dataset)}")

    print(f"Test Samples              : " f"{len(test_loader.dataset)}")

    print(f"Quantized Modules         : " f"{quantized_module_count}")

    print()

    print(f"Initial PyTorch PTQ Acc.  : " f"{ptq_accuracy * 100:.2f}%")

    print(
        f"Paired PyTorch PTQ Acc.   : " f"{fidelity_results['pytorch_accuracy']:.2f}%"
    )

    print(f"Exported ONNX Accuracy    : " f"{fidelity_results['onnx_accuracy']:.2f}%")

    print(
        f"Export Accuracy Change    : "
        f"{fidelity_results['accuracy_difference']:+.2f} pp"
    )

    print(
        f"Prediction Agreement      : "
        f"{fidelity_results['prediction_agreement']:.2f}%"
    )

    print()

    print(f"ONNX File Size            : " f"{model_size_mb:.2f} MB")

    print(f"ONNX File                 : " f"{args.output}")

    print("ONNX Valid                : YES")

    print("========================================")


if __name__ == "__main__":
    main()
