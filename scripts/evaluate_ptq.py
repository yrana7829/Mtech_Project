import sys
import os
import torch
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.evaluation.evaluate import evaluate
from src.models.model_loader import get_model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    device = torch.device("cpu")

    print("Loading dataset...")
    _, _, test_loader = get_dataset(args.dataset)

    print("Loading model architecture...")
    model = get_model(args.model, num_classes=10)

    print("Loading checkpoint...")
    state_dict = torch.load(args.checkpoint, map_location=device)

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print("Running evaluation...\n")

    acc = evaluate(model, test_loader, device)

    print(f"\nPTQ Accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
