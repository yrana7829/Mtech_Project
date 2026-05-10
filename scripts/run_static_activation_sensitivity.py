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


CALIB_SIZE = 1000
CALIB_SEED = 42

torch.manual_seed(CALIB_SEED)
np.random.seed(CALIB_SEED)
random.seed(CALIB_SEED)


def get_target_blocks(model_name, model):

    target_blocks = []

    if model_name.lower() == "mobilenetv2":

        for name, module in model.named_modules():

            if "features." in name and name.count(".") == 1:
                target_blocks.append(name)

    elif model_name.lower() == "resnet18":

        for name, module in model.named_modules():

            if "layer" in name and name.count(".") == 1:
                target_blocks.append(name)

    return target_blocks


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

    print(f"Calibration samples: {len(calib_dataset)}")

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

    baseline_acc = baseline_acc * 100

    print(f"FP32 Accuracy: {baseline_acc:.2f}%")

    # -----------------------------------
    # Calibration
    # -----------------------------------
    print("\nCollecting activation ranges...")

    activation_stats = collect_activation_ranges(model, calib_loader)

    # -----------------------------------
    # Target blocks
    # -----------------------------------
    target_blocks = get_target_blocks(args.model, model)

    print("\n[INFO] Target Blocks:")

    for block in target_blocks:
        print(block)

    results = []

    # -----------------------------------
    # Layer-wise sensitivity
    # -----------------------------------
    for block_name in target_blocks:

        print("\n====================================")
        print(f"[RUNNING] Block: {block_name}")
        print("====================================")

        analyzer = StaticActivationSensitivityAnalyzer(
            model=model,
            activation_stats=activation_stats,
            target_module_names=[block_name],
        )

        analyzer.register_hooks()

        quant_acc = evaluate(model, val_loader, device)

        quant_acc = quant_acc * 100

        analyzer.remove_hooks()

        drop = baseline_acc - quant_acc

        print(f"\n[RESULT] Accuracy: {quant_acc:.2f}%")
        print(f"[DROP] {drop:.2f}%")

        results.append(
            {
                "block": block_name,
                "fp32_acc": baseline_acc,
                "quant_acc": quant_acc,
                "acc_drop": drop,
            }
        )

    # -----------------------------------
    # Save
    # -----------------------------------
    output_dir = "results/Phase4_ActivationSensitivity"

    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(
        output_dir, f"{args.model}_{args.dataset}_static_activation_sensitivity.csv"
    )

    df = pd.DataFrame(results)

    df.to_csv(save_path, index=False)

    print(f"\n[INFO] Results saved to: {save_path}")


if __name__ == "__main__":
    main()
