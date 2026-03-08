import sys
import os
import torch
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.evaluation.evaluate import evaluate


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--quant_checkpoint", required=True)

    args = parser.parse_args()

    device = torch.device("cpu")

    print("Loading dataset...")
    _, _, test_loader = get_dataset(args.dataset)

    print("Loading quantized model...")

    # IMPORTANT FIX
    model = torch.load(args.quant_checkpoint, map_location="cpu", weights_only=False)

    model.eval()
    model.to(device)

    print("\nEvaluating quantized model...\n")

    acc = evaluate(model, test_loader, device)

    print(f"Quantized Accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
