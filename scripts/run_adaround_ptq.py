import argparse
import torch
import os
import sys

# Add project root
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
    train_loader, _, test_loader = get_dataset(args.dataset)

    print("Loading model...")
    model = get_model(args.model, num_classes=10)

    print("Loading FP32 checkpoint...")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # Evaluate FP32
    print("\nEvaluating FP32 model...")
    fp32_acc = evaluate(model, test_loader, device)
    print(f"FP32 Accuracy: {fp32_acc*100:.2f}%")

    # Apply AdaRound PTQ
    print("\nApplying AdaRound PTQ...")
    model = apply_adaround(model, train_loader, device)

    # Debug: show sample quantized weights
    print("\nSample quantized weight values:")

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            print(m.weight.view(-1)[:5])
            break

    # Evaluate quantized model
    print("\nEvaluating quantized model...")
    acc = evaluate(model, test_loader, device)

    print(f"\nAdaRound Accuracy: {acc*100:.2f}%")

    # Save results
    os.makedirs("results/ptq_results", exist_ok=True)

    with open("results/ptq_results/adaround_results.txt", "a") as f:
        f.write(f"{args.dataset},{args.model},{acc*100:.2f}\n")


if __name__ == "__main__":
    main()
