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

from src.quantization.proposed.proposed_ptq_pipeline import (
    apply_proposed_ptq_pipeline,
)

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

torch.backends.quantized.engine = "fbgemm"


# ============================================================
# SET ALL RANDOM SEEDS
# ============================================================


def set_all_seeds(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ============================================================
# CREATE FIXED CALIBRATION INDICES
# ============================================================


def create_fixed_calibration_indices(train_dataset):

    generator = torch.Generator()

    generator.manual_seed(CALIB_SEED)

    indices = torch.randperm(len(train_dataset), generator=generator)[:CALIB_SIZE]

    return indices


# ============================================================
# BUILD CALIBRATION LOADER
# ============================================================


def build_calibration_loader(train_dataset, fixed_indices, batch_size):

    calib_dataset = Subset(train_dataset, fixed_indices)

    calib_loader = DataLoader(
        calib_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return calib_loader


# ============================================================
# BUILD DETERMINISTIC NAIVE FX PTQ MODEL
# ============================================================


def build_naive_ptq_model(fp32_model, calibration_loader):

    device = torch.device("cpu")

    fp32_model.eval()
    fp32_model.to(device)

    # --------------------------------------------------------
    # QUANTIZATION CONFIGURATION
    # --------------------------------------------------------

    qconfig = get_default_qconfig("fbgemm")

    qconfig_dict = {"": qconfig}

    # --------------------------------------------------------
    # EXAMPLE INPUT
    #
    # This accesses the training dataset, which contains:
    #
    # RandomHorizontalFlip
    # RandomRotation(10)
    #
    # Therefore this step consumes random state.
    #
    # That is acceptable because we reset every RNG
    # immediately before actual calibration.
    # --------------------------------------------------------

    example_inputs = next(iter(calibration_loader))[0][:1].to(device)

    print("\nPreparing FX PTQ model...")

    prepared_model = prepare_fx(fp32_model, qconfig_dict, example_inputs)

    # --------------------------------------------------------
    # CRITICAL REPRODUCIBILITY STEP
    #
    # The calibration dataset uses random augmentation.
    #
    # Therefore reset all RNGs immediately before the
    # actual calibration pass.
    #
    # Combined with:
    #
    # fixed image indices
    # shuffle=False
    # num_workers=0
    #
    # this reproduces the exact same calibration tensors
    # across independent PTQ reconstructions.
    # --------------------------------------------------------

    print("\nResetting RNG state before calibration...")

    set_all_seeds(CALIB_SEED)

    # --------------------------------------------------------
    # ACTUAL CALIBRATION
    # --------------------------------------------------------

    print("Running deterministic calibration...")

    with torch.no_grad():

        for images, _ in calibration_loader:

            images = images.to(device)

            prepared_model(images)

    # --------------------------------------------------------
    # CONVERT TO REAL INT8 MODEL
    # --------------------------------------------------------

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
# EXTRACT QUANTIZATION SIGNATURE
# ============================================================


def extract_quantization_signature(quantized_model):

    signature = []

    for name, module in quantized_model.named_modules():

        scale = None
        zero_point = None

        # ----------------------------------------------------
        # SCALE
        # ----------------------------------------------------

        if hasattr(module, "scale"):

            try:

                scale = module.scale

                if torch.is_tensor(scale):

                    scale = scale.detach().cpu().numpy().tolist()

                else:

                    scale = float(scale)

            except Exception:

                scale = str(module.scale)

        # ----------------------------------------------------
        # ZERO POINT
        # ----------------------------------------------------

        if hasattr(module, "zero_point"):

            try:

                zero_point = module.zero_point

                if torch.is_tensor(zero_point):

                    zero_point = zero_point.detach().cpu().numpy().tolist()

                else:

                    zero_point = int(zero_point)

            except Exception:

                zero_point = str(module.zero_point)

        # ----------------------------------------------------
        # RECORD QUANTIZED MODULE PARAMETERS
        # ----------------------------------------------------

        if scale is not None or zero_point is not None:

            signature.append((name, scale, zero_point))

    return signature


# ============================================================
# EXPORT TO ONNX
# ============================================================


def export_to_onnx(model, output_path):

    output_directory = os.path.dirname(output_path)

    if output_directory:

        os.makedirs(output_directory, exist_ok=True)

    # --------------------------------------------------------
    # FIX EXPORT INPUT SEED
    #
    # The dummy input does not affect calibration because
    # quantization is already complete.
    #
    # We still fix the seed for complete reproducibility.
    # --------------------------------------------------------

    torch.manual_seed(GLOBAL_SEED)

    dummy_input = torch.randn(1, 3, 224, 224)

    print("\nExporting deterministic " "Naive FX PTQ model to ONNX...")

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
# PAIRED PYTORCH PTQ VS EXPORTED ONNX COMPARISON
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

    print("\nComparing exact deterministic " "PyTorch PTQ model with exported ONNX...")

    with torch.no_grad():

        for images, labels in test_loader:

            batch_size = images.size(0)

            for i in range(batch_size):

                # ============================================
                # ONE TEST IMAGE
                # ============================================

                image = images[i : i + 1]

                label = int(labels[i].item())

                # ============================================
                # EXACT PYTORCH PTQ MODEL
                # ============================================

                pytorch_output = quant_model(image)

                pytorch_output_np = pytorch_output.detach().cpu().numpy()

                pytorch_pred = int(np.argmax(pytorch_output_np, axis=1)[0])

                # ============================================
                # EXACT EXPORTED ONNX MODEL
                # ============================================

                onnx_input = image.cpu().numpy().astype(np.float32)

                onnx_output = session.run(None, {input_name: onnx_input})[0]

                onnx_pred = int(np.argmax(onnx_output, axis=1)[0])

                # ============================================
                # ACCURACY
                # ============================================

                pytorch_is_correct = pytorch_pred == label

                onnx_is_correct = onnx_pred == label

                if pytorch_is_correct:

                    pytorch_correct += 1

                if onnx_is_correct:

                    onnx_correct += 1

                # ============================================
                # PREDICTION AGREEMENT
                # ============================================

                if pytorch_pred == onnx_pred:

                    prediction_matches += 1

                # ============================================
                # DIRECTION OF CHANGED DECISIONS
                # ============================================

                if pytorch_is_correct and not onnx_is_correct:

                    pytorch_correct_onnx_wrong += 1

                if not pytorch_is_correct and onnx_is_correct:

                    pytorch_wrong_onnx_correct += 1

                # ============================================
                # NUMERICAL OUTPUT DIFFERENCE
                # ============================================

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

    average_mean_difference = float(np.mean(mean_differences))

    average_max_difference = float(np.mean(max_differences))

    worst_max_difference = float(np.max(max_differences))

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

    print(f"Average Mean Output Difference     : " f"{average_mean_difference:.6f}")

    print(f"Average Max Output Difference      : " f"{average_max_difference:.6f}")

    print(f"Worst Max Output Difference        : " f"{worst_max_difference:.6f}")

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
        "average_mean_difference": average_mean_difference,
        "average_max_difference": average_max_difference,
        "worst_max_difference": worst_max_difference,
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
    # RESET GLOBAL STATE
    # ========================================================

    set_all_seeds(GLOBAL_SEED)

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print("Loading dataset...")

    train_loader, val_loader, test_loader = get_dataset(args.dataset)

    train_dataset = train_loader.dataset

    # ========================================================
    # FIX CALIBRATION INDICES
    # ========================================================

    fixed_indices = create_fixed_calibration_indices(train_dataset)

    calibration_loader = build_calibration_loader(
        train_dataset=train_dataset,
        fixed_indices=fixed_indices,
        batch_size=train_loader.batch_size,
    )

    print(f"Training samples: " f"{len(train_dataset)}")

    print(f"Calibration samples: " f"{len(calibration_loader.dataset)}")

    print(f"Calibration seed: " f"{CALIB_SEED}")

    print(f"Test samples: " f"{len(test_loader.dataset)}")

    # ========================================================
    # LOAD FRESH FP32 MODEL
    # ========================================================

    print("\nLoading FP32 model...")

    fp32_model = get_model(args.model, num_classes=args.num_classes)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    fp32_model.load_state_dict(checkpoint)

    fp32_model.eval()

    # ========================================================
    # BUILD DETERMINISTIC NAIVE FX PTQ MODEL
    # ========================================================

    print("\nApplying PTQ++ pipeline...")

    proposed_model = apply_proposed_ptq_pipeline(fp32_model, device)

    print("\nRunning FX quantization on PTQ++ model...")

    quant_model = build_naive_ptq_model(proposed_model, calibration_loader)

    # ========================================================
    # INSPECT QUANTIZED MODEL
    # ========================================================

    quantized_module_count = inspect_quantized_modules(quant_model)

    # ========================================================
    # RECORD QUANTIZATION SIGNATURE
    # ========================================================

    quantization_signature = extract_quantization_signature(quant_model)

    print(f"\nQuantization Parameters Recorded: " f"{len(quantization_signature)}")

    # ========================================================
    # EVALUATE SOURCE PYTORCH PTQ MODEL
    # ========================================================

    print("\nEvaluating deterministic " "PyTorch FX PTQ model on TEST SET...")

    ptq_accuracy = evaluate(quant_model, test_loader, device)

    ptq_accuracy_percent = ptq_accuracy * 100.0

    print(
        f"\nDeterministic PyTorch FX PTQ "
        f"Test Accuracy: "
        f"{ptq_accuracy_percent:.2f}%"
    )

    # ========================================================
    # EXPORT EXACT MODEL JUST EVALUATED
    # ========================================================

    export_to_onnx(quant_model, args.output)

    # ========================================================
    # STRUCTURAL VALIDATION
    # ========================================================

    op_counts = validate_onnx_structure(args.output)

    # ========================================================
    # FILE SIZE
    # ========================================================

    model_size_mb = os.path.getsize(args.output) / (1024**2)

    print(f"\nONNX File Size: " f"{model_size_mb:.2f} MB")

    # ========================================================
    # PAIRED EXPORT FIDELITY TEST
    #
    # Exact source object
    # vs
    # Exact exported artifact
    # ========================================================

    fidelity_results = compare_pytorch_ptq_and_onnx(
        quant_model=quant_model, onnx_path=args.output, test_loader=test_loader
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n\n========================================")

    print("FINAL DETERMINISTIC NAIVE PTQ " "EXPORT SUMMARY")

    print("========================================")

    print("Method:")

    print("Naive FX PTQ INT8")

    print()

    print(f"Calibration Samples       : " f"{len(calibration_loader.dataset)}")

    print(f"Calibration Seed          : " f"{CALIB_SEED}")

    print(f"Test Samples              : " f"{len(test_loader.dataset)}")

    print(f"Quantized Modules         : " f"{quantized_module_count}")

    print(f"Quantization Signatures   : " f"{len(quantization_signature)}")

    print()

    print(f"Source PyTorch PTQ Acc.   : " f"{ptq_accuracy_percent:.2f}%")

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

    print(
        f"Changed Predictions       : "
        f"{fidelity_results['changed_predictions']} / "
        f"{fidelity_results['total']}"
    )

    print()

    print(f"ONNX Nodes                : " f"{sum(op_counts.values())}")

    print(f"ONNX File Size            : " f"{model_size_mb:.2f} MB")

    print(f"ONNX File                 : " f"{args.output}")

    print("ONNX Valid                : YES")

    print("========================================")


if __name__ == "__main__":

    main()
