# scripts/run_adaround_ptq.py

import argparse
import torch
import os
import sys

# Ensure project root is available for imports

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate
from src.quantization.ptq.adaround_ptq import apply_adaround


def main():

    parser = argparse.ArgumentParser(description="Run AdaRound PTQ experiment")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    train_loader, _, test_loader = get_dataset(args.dataset)

    print("Loading model...")
    model = get_model(args.model, num_classes=10)
    model = model.to(device)

    print("Loading FP32 checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)

    model.eval()

    print("Applying AdaRound PTQ...")
    model = apply_adaround(model, train_loader, device)

    print("\nEvaluating quantized model...")
    acc = evaluate(model, test_loader, device)

    print(f"\nAdaRound Accuracy: {acc:.2f}%")

    # Save experiment results
    os.makedirs("results/ptq_results", exist_ok=True)

    results_file = "results/ptq_results/adaround_ptq_results.txt"

    with open(results_file, "a") as f:
        f.write(f"{args.dataset},{args.model},{acc:.2f}\n")


if __name__ == "__main__":
    main()
