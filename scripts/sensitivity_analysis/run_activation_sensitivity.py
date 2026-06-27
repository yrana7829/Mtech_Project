import argparse
import os
import sys

import pandas as pd
import torch

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate

from src.quantization.sensitivity.activation_sensitivity import (
    ActivationSensitivityAnalyzer,
)


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

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------
    # Load dataset
    # -----------------------------------
    print("Loading dataset...")
    train_loader, _, test_loader = get_dataset(args.dataset)

    # -----------------------------------
    # Load model
    # -----------------------------------
    print("Loading model...")
    model = get_model(args.model, num_classes=10)

    # -----------------------------------
    # Load checkpoint
    # -----------------------------------
    print("Loading checkpoint...")

    state_dict = torch.load(args.checkpoint, map_location=device)

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # -----------------------------------
    # Baseline evaluation
    # -----------------------------------
    print("\nEvaluating FP32 model...")

    baseline_acc = evaluate(model, test_loader, device)

    baseline_acc = baseline_acc * 100

    print(f"[BASELINE FP32 ACC]: {baseline_acc:.2f}%")

    # -----------------------------------
    # Get target blocks
    # -----------------------------------
    target_blocks = get_target_blocks(args.model, model)

    print("\n[INFO] Target Blocks:")

    for block in target_blocks:
        print(block)

    results = []

    # -----------------------------------
    # Sensitivity analysis
    # -----------------------------------
    for block_name in target_blocks:

        print(f"\n[RUNNING] Quantizing activations for block: {block_name}")

        analyzer = ActivationSensitivityAnalyzer(
            model=model, target_module_names=[block_name]
        )

        analyzer.register_hooks()

        quant_acc = evaluate(model, test_loader, device)

        quant_acc = quant_acc * 100

        analyzer.remove_hooks()

        drop = baseline_acc - quant_acc

        print(f"[RESULT] Accuracy: {quant_acc:.2f}%")
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
    # Save results
    # -----------------------------------
    output_dir = "results/Phase4_ActivationSensitivity"

    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(
        output_dir, f"{args.model}_{args.dataset}_activation_sensitivity.csv"
    )

    df = pd.DataFrame(results)

    df.to_csv(save_path, index=False)

    print(f"\n[INFO] Results saved to: {save_path}")


if __name__ == "__main__":
    main()
