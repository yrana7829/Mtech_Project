import sys
import os
import torch
import argparse

print("Script started")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("Imports path set")

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.quantization.ptq.naive_ptq import naive_ptq
from src.evaluation.evaluate import evaluate


def main():

    print("Entering main()")

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    device = torch.device("cpu")

    train_loader, _, test_loader = get_dataset(args.dataset)

    print("Dataset loaded")

    model = get_model(args.model, num_classes=10)

    print("Model loaded")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    print("Checkpoint loaded")

    print("Running naive PTQ...")

    quant_model = naive_ptq(model, train_loader, device)

    print("PTQ finished")

    print("Evaluating quantized model...")

    acc = evaluate(quant_model, test_loader, torch.device("cpu"))

    print("Quantized accuracy:", acc * 100)


if __name__ == "__main__":
    main()
