# scripts/run_adaround_ptq.py

import argparse
import torch
import os
import sys

# 1. Ensure path is correct for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate
from src.quantization.ptq.adaround_ptq import apply_adaround


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    args = parser.parse_args()

    # AdaRound usually benefits from GPU for the optimization phase
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    # 2. Fix: Added '_' to unpack 3 values
    train_loader, _, test_loader = get_dataset(args.dataset)

    print("Loading model...")
    # 3. Fix: Arguments must match your get_model definition (num_classes)
    model = get_model(args.model, num_classes=10)

    print("Loading FP32 checkpoint...")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    print("Applying AdaRound PTQ...")
    # AdaRound optimizes rounding weights per layer
    model = apply_adaround(model, train_loader, device)

    print("\nEvaluating quantized model...")
    # 4. Consistency: Match the evaluation call style from naive script
    acc = evaluate(model, test_loader, device)

    # If your evaluate returns a decimal (0.85), use *100.
    # If it returns percentage (85.0), remove the *100.
    print(f"\nAdaRound Accuracy: {acc*100:.2f}%")

    # Optional: Save results to disk like the naive script
    os.makedirs("results/ptq_results", exist_ok=True)
    with open("results/ptq_results/adaround_ptq_results.txt", "a") as f:
        f.write(f"{args.dataset},{args.model},{acc*100:.2f}\n")


if __name__ == "__main__":
    main()
