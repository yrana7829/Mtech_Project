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
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    device = torch.device("cpu")

    print("Loading dataset...")
    _, _, test_loader = get_dataset(args.dataset)

    print("Loading quantized model...")

    model = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,  # required for PyTorch 2.6+
    )

    model.eval()

    print("Running evaluation...\n")

    acc = evaluate(model, test_loader, device)

    print(f"\nPTQ Accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
