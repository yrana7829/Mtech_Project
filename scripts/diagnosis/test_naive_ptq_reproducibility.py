import argparse
import os
import sys
import random

import numpy as np
import torch

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
# EXPERIMENT CONFIGURATION
# ============================================================

CALIB_SIZE = 1000
CALIB_SEED = 42

GLOBAL_SEED = 42

NUM_RUNS = 3


# ============================================================
# SET ALL RANDOM SEEDS
# ============================================================


def set_all_seeds(seed):

    torch.manual_seed(seed)

    np.random.seed(seed)

    random.seed(seed)


# ============================================================
# BUILD FIXED CALIBRATION INDICES
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
# BUILD ONE PTQ MODEL
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
    # CREATE EXAMPLE INPUT
    #
    # This accesses the augmented training dataset.
    # Therefore it is done BEFORE resetting the calibration RNG.
    # --------------------------------------------------------

    example_inputs = next(iter(calibration_loader))[0][:1].to(device)

    # --------------------------------------------------------
    # PREPARE FX MODEL
    # --------------------------------------------------------

    prepared_model = prepare_fx(fp32_model, qconfig_dict, example_inputs)

    # --------------------------------------------------------
    # CRITICAL STEP
    #
    # Reset every RNG immediately before actual calibration.
    #
    # Therefore every run should see:
    #
    # Same images
    # Same image order
    # Same horizontal flips
    # Same rotation angles
    # --------------------------------------------------------

    set_all_seeds(CALIB_SEED)

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    with torch.no_grad():

        for images, _ in calibration_loader:

            images = images.to(device)

            prepared_model(images)

    # --------------------------------------------------------
    # CONVERT TO REAL INT8 MODEL
    # --------------------------------------------------------

    quantized_model = convert_fx(prepared_model)

    quantized_model.eval()

    return quantized_model


# ============================================================
# EXTRACT QUANTIZATION PARAMETERS
#
# Used as a second reproducibility check.
# If PTQ is reproducible, these should also match.
# ============================================================


def extract_quantization_signature(quantized_model):

    signature = []

    for name, module in quantized_model.named_modules():

        # ----------------------------------------------------
        # OUTPUT SCALE
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

        else:

            scale = None

        # ----------------------------------------------------
        # OUTPUT ZERO POINT
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

        else:

            zero_point = None

        # ----------------------------------------------------
        # SAVE ONLY MODULES WITH QUANTIZATION PARAMETERS
        # ----------------------------------------------------

        if scale is not None or zero_point is not None:

            signature.append((name, scale, zero_point))

    return signature


# ============================================================
# COMPARE QUANTIZATION SIGNATURES
# ============================================================


def compare_signatures(signature_a, signature_b):

    return signature_a == signature_b


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)

    parser.add_argument("--model", required=True)

    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--runs", type=int, default=NUM_RUNS)

    args = parser.parse_args()

    device = torch.device("cpu")

    # ========================================================
    # LOAD DATASET ONCE
    # ========================================================

    print("\nLoading dataset...")

    train_loader, val_loader, test_loader = get_dataset(args.dataset)

    train_dataset = train_loader.dataset

    print(f"Training samples: " f"{len(train_dataset)}")

    print(f"Test samples: " f"{len(test_loader.dataset)}")

    # ========================================================
    # CREATE FIXED CALIBRATION INDICES ONCE
    # ========================================================

    fixed_indices = create_fixed_calibration_indices(train_dataset)

    print(f"Calibration samples: " f"{len(fixed_indices)}")

    print(f"Calibration seed: " f"{CALIB_SEED}")

    print(f"Number of PTQ runs: " f"{args.runs}")

    # ========================================================
    # STORAGE
    # ========================================================

    run_accuracies = []

    run_signatures = []

    # ========================================================
    # REPEATED PTQ RECONSTRUCTION
    # ========================================================

    for run_number in range(1, args.runs + 1):

        print("\n\n" "========================================")

        print(f"PTQ REPRODUCIBILITY RUN " f"{run_number}/{args.runs}")

        print("========================================")

        # ----------------------------------------------------
        # RESET GLOBAL STATE BEFORE EACH COMPLETE RUN
        # ----------------------------------------------------

        set_all_seeds(GLOBAL_SEED)

        # ----------------------------------------------------
        # BUILD FRESH CALIBRATION LOADER
        # ----------------------------------------------------

        calibration_loader = build_calibration_loader(
            train_dataset=train_dataset,
            fixed_indices=fixed_indices,
            batch_size=train_loader.batch_size,
        )

        # ----------------------------------------------------
        # LOAD A COMPLETELY FRESH FP32 MODEL
        #
        # Important:
        # Never reuse a model already passed through prepare_fx.
        # ----------------------------------------------------

        print("\nLoading fresh FP32 model...")

        fp32_model = get_model(args.model, num_classes=args.num_classes)

        checkpoint = torch.load(args.checkpoint, map_location=device)

        fp32_model.load_state_dict(checkpoint)

        fp32_model.eval()

        # ----------------------------------------------------
        # BUILD PTQ MODEL
        # ----------------------------------------------------

        print("Preparing and calibrating PTQ model...")

        quantized_model = build_naive_ptq_model(
            fp32_model=fp32_model, calibration_loader=calibration_loader
        )

        # ----------------------------------------------------
        # EVALUATE ON SAME TEST SET
        # ----------------------------------------------------

        print("Evaluating on test set...")

        accuracy = evaluate(quantized_model, test_loader, device)

        accuracy_percent = accuracy * 100.0

        run_accuracies.append(accuracy_percent)

        # ----------------------------------------------------
        # EXTRACT QUANTIZATION SIGNATURE
        # ----------------------------------------------------

        signature = extract_quantization_signature(quantized_model)

        run_signatures.append(signature)

        print(f"\nRUN {run_number} RESULT")

        print(f"Test Accuracy: " f"{accuracy_percent:.4f}%")

        print(f"Quantization Parameters Recorded: " f"{len(signature)}")

        # ----------------------------------------------------
        # CLEAN UP
        # ----------------------------------------------------

        del fp32_model

        del quantized_model

    # ========================================================
    # FINAL REPRODUCIBILITY ANALYSIS
    # ========================================================

    print("\n\n" "========================================")

    print("FINAL PTQ REPRODUCIBILITY RESULT")

    print("========================================")

    for run_number, accuracy in enumerate(run_accuracies, start=1):

        print(f"Run {run_number} Accuracy: " f"{accuracy:.4f}%")

    print()

    # --------------------------------------------------------
    # ACCURACY REPRODUCIBILITY
    # --------------------------------------------------------

    accuracy_identical = all(
        accuracy == run_accuracies[0] for accuracy in run_accuracies
    )

    print(f"All Accuracies Identical: " f"{accuracy_identical}")

    # --------------------------------------------------------
    # QUANTIZATION PARAMETER REPRODUCIBILITY
    # --------------------------------------------------------

    signatures_identical = all(
        compare_signatures(run_signatures[0], signature)
        for signature in run_signatures[1:]
    )

    print(f"All Quantization Signatures Identical: " f"{signatures_identical}")

    # --------------------------------------------------------
    # ACCURACY STATISTICS
    # --------------------------------------------------------

    mean_accuracy = float(np.mean(run_accuracies))

    std_accuracy = (
        float(np.std(run_accuracies, ddof=1)) if len(run_accuracies) > 1 else 0.0
    )

    min_accuracy = float(np.min(run_accuracies))

    max_accuracy = float(np.max(run_accuracies))

    accuracy_range = max_accuracy - min_accuracy

    print()

    print(f"Mean Accuracy: " f"{mean_accuracy:.4f}%")

    print(f"Std Accuracy: " f"{std_accuracy:.4f} pp")

    print(f"Minimum Accuracy: " f"{min_accuracy:.4f}%")

    print(f"Maximum Accuracy: " f"{max_accuracy:.4f}%")

    print(f"Accuracy Range: " f"{accuracy_range:.4f} pp")

    print("========================================")

    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    if accuracy_identical and signatures_identical:

        print("\nRESULT:")

        print("PTQ reconstruction is reproducible.")

        print(
            "The same calibration procedure produced "
            "identical accuracy and identical "
            "quantization parameters in every run."
        )

    elif accuracy_identical:

        print("\nRESULT:")

        print("Accuracy is reproducible, but some " "quantization parameters differ.")

        print(
            "Further investigation is required before " "freezing the deployment model."
        )

    else:

        print("\nRESULT:")

        print("PTQ reconstruction is still not reproducible.")

        print("Do not proceed to final ONNX export yet.")


if __name__ == "__main__":

    main()
