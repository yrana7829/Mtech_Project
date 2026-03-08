# scripts/run_adaround_ptq.py

import argparse
import torch

from src.dataset.dataloader import get_dataloader
from src.models.model_loader import load_model
from src.evaluation.evaluate import evaluate_model
from src.quantization.ptq.adaround_ptq import apply_adaround


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    train_loader, test_loader = get_dataloader(args.dataset)

    print("Loading model...")
    model = load_model(args.model, args.dataset)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    print("Applying AdaRound PTQ...")
    model = apply_adaround(model, train_loader, device)

    print("Evaluating quantized model...")
    acc = evaluate_model(model, test_loader, device)

    print(f"AdaRound Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
