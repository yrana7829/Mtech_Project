import sys
import os
import torch
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    _, _, test_loader = get_dataset(args.dataset)

    model = get_model(args.model, num_classes=10)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    acc = evaluate(model, test_loader, device)

    print(f"\nTest Accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
