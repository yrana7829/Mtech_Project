import argparse
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate
from src.quantization.proposed.mixed_precision import apply_mixed_precision
from src.quantization.proposed.naive_proposed_ptq import apply_naive_ptq


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    _, _, test_loader = get_dataset(args.dataset)

    print("Loading model...")
    model = get_model(args.model, num_classes=10)

    print("Loading checkpoint...")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print("\nEvaluating FP32...")
    fp32 = evaluate(model, test_loader, device)
    print(f"FP32 Accuracy: {fp32*100:.2f}%")

    print("\nApplying Mixed Precision Allocation...")
    model = apply_mixed_precision(model)

    print("\nEvaluating quantized model...")
    acc = evaluate(model, test_loader, device)

    print(f"\nMPA Accuracy: {acc*100:.2f}%")

    os.makedirs("results/proposed_results", exist_ok=True)

    with open("results/proposed_results/mpa_results.txt", "a") as f:
        f.write(f"{args.dataset},{args.model},{acc*100:.2f}\n")


if __name__ == "__main__":
    main()
