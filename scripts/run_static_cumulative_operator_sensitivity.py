import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
import random

from torch.utils.data import Subset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate

from src.quantization.sensitivity.static_activation_sensitivity import (
    collect_activation_ranges,
    StaticActivationSensitivityAnalyzer,
)

# -----------------------------------
# Config
# -----------------------------------
CALIB_SIZE = 1000
CALIB_SEED = 42

torch.manual_seed(CALIB_SEED)
np.random.seed(CALIB_SEED)
random.seed(CALIB_SEED)


# -----------------------------------
# Target operators
# -----------------------------------
def get_target_operators(model):

    target_ops = []

    for name, module in model.named_modules():

        if isinstance(module, (torch.nn.Conv2d, torch.nn.ReLU, torch.nn.ReLU6)):

            target_ops.append(name)

    return target_ops


# -----------------------------------
# Main
# -----------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    device = torch.device("cpu")

    # -----------------------------------
    # Dataset
    # -----------------------------------
    print("Loading dataset...")

    train_loader, val_loader, _ = get_dataset(args.dataset)

    train_dataset = train_loader.dataset

    indices = torch.randperm(len(train_dataset))[:CALIB_SIZE]

    calib_dataset = Subset(train_dataset, indices)

    calib_loader = DataLoader(calib_dataset, batch_size=16, shuffle=False)

    print(f"Calibration samples: " f"{len(calib_dataset)}")

    # -----------------------------------
    # Model
    # -----------------------------------
    print("Loading model...")

    model = get_model(args.model, num_classes=10)

    print("Loading checkpoint...")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    model.eval()

    # -----------------------------------
    # Baseline
    # -----------------------------------
    print("\nEvaluating FP32 model...")

    baseline_acc = evaluate(model, val_loader, device)

    baseline_acc *= 100

    print(f"FP32 Accuracy: " f"{baseline_acc:.2f}%")

    # -----------------------------------
    # Collect ranges
    # -----------------------------------
    print("\nCollecting activation ranges...")

    activation_stats = collect_activation_ranges(model, calib_loader)

    # -----------------------------------
    # Target operators
    # -----------------------------------
    target_ops = get_target_operators(model)

    print("\n[INFO] Target Operators:")

    for op in target_ops:
        print(op)

    results = []

    # -----------------------------------
    # Progressive cumulative sensitivity
    # -----------------------------------
    for i in range(len(target_ops)):

        current_ops = target_ops[: i + 1]

        print("\n" + "=" * 60)
        print(f"[RUNNING] " f"Quantizing first " f"{len(current_ops)} operators")

        print(f"Last operator: " f"{current_ops[-1]}")

        print("=" * 60)

        analyzer = StaticActivationSensitivityAnalyzer(
            model=model,
            activation_stats=activation_stats,
            target_module_names=current_ops,
        )

        analyzer.register_hooks()

        quant_acc = evaluate(model, val_loader, device)

        quant_acc *= 100

        analyzer.remove_hooks()

        drop = baseline_acc - quant_acc

        print(f"\n[RESULT] " f"Accuracy: " f"{quant_acc:.2f}%")

        print(f"[DROP] " f"{drop:.2f}%")

        results.append(
            {
                "num_quantized_operators": len(current_ops),
                "last_operator": current_ops[-1],
                "fp32_acc": baseline_acc,
                "quant_acc": quant_acc,
                "acc_drop": drop,
            }
        )

    # -----------------------------------
    # Save
    # -----------------------------------
    output_dir = "results/" "Phase4_ActivationSensitivity"

    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(
        output_dir,
        f"{args.model}_"
        f"{args.dataset}_"
        f"static_cumulative_"
        f"operator_sensitivity.csv",
    )

    df = pd.DataFrame(results)

    df.to_csv(save_path, index=False)

    print(f"\n[INFO] Results saved to:\n" f"{save_path}")


if __name__ == "__main__":
    main()
