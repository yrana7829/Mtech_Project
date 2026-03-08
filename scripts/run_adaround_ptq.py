# scripts/run_adaround_ptq.py

import argparse
import torch
import os
import sys

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    train_loader, test_loader = get_dataset(args.dataset)

    print("Loading model...")
    model = get_model(args.model, args.dataset)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    print("Applying AdaRound PTQ...")
    model = apply_adaround(model, train_loader, device)

    print("Evaluating quantized model...")
    acc = evaluate(model, test_loader, device)

    print(f"AdaRound Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
