import os
import argparse
import pandas as pd
import torch

from src.models.model_loader import get_model
from src.dataset.dataloader import get_dataset
from src.evaluation.evaluate import evaluate

from src.quantization.sensitivity.activation_sensitivity import (
    ActivationSensitivityAnalyzer,
)


def get_target_blocks(model_name, model):

    target_blocks = []

    if model_name.lower() == "mobilenetv2":

        for name, module in model.named_modules():

            # Inverted residual blocks
            if "features." in name and name.count(".") == 1:
                target_blocks.append(name)

    elif model_name.lower() == "resnet18":

        for name, module in model.named_modules():

            if "layer" in name and name.count(".") == 1:
                target_blocks.append(name)

    return target_blocks


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Load model
    # -------------------------
    model = get_model(
        model_name=args.model,
        num_classes=args.num_classes,
        checkpoint_path=args.checkpoint,
    )

    model.to(device)
    model.eval()

    # -------------------------
    # Load dataset
    # -------------------------
    _, _, test_loader = get_dataset(
        dataset_name=args.dataset, batch_size=args.batch_size
    )

    # -------------------------
    # Baseline accuracy
    # -------------------------
    baseline_acc = evaluate(model, test_loader, device)

    print(f"\n[BASELINE FP32 ACC]: {baseline_acc:.2f}%")

    # -------------------------
    # Target blocks
    # -------------------------
    target_blocks = get_target_blocks(args.model, model)

    print("\n[INFO] Target Blocks:")
    for block in target_blocks:
        print(block)

    results = []

    # -------------------------
    # Sensitivity loop
    # -------------------------
    for block_name in target_blocks:

        print(f"\n[RUNNING] Quantizing activations for block: {block_name}")

        analyzer = ActivationSensitivityAnalyzer(
            model=model, target_module_names=[block_name]
        )

        analyzer.register_hooks()

        quant_acc = evaluate(model, test_loader, device)

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

    # -------------------------
    # Save results
    # -------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    save_path = os.path.join(
        args.output_dir, f"{args.model}_{args.dataset}_activation_sensitivity.csv"
    )

    df = pd.DataFrame(results)

    df.to_csv(save_path, index=False)

    print(f"\n[INFO] Results saved to: {save_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--num_classes", type=int, required=True)

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--output_dir", type=str, default="results/Phase4_ActivationSensitivity/"
    )

    args = parser.parse_args()

    main(args)
