import sys
import os
from xml.parsers.expat import model
import torch
import argparse
import numpy as np
import random
from torch.utils.data import Subset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate
from src.quantization.ptq.adaround_minimal import apply_adaround, fx_quantize_model


# -------------------------------
# Config
# -------------------------------
CALIB_SIZE = 1000
CALIB_SEED = 42

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# -------------------------------
# Main
# -------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    train_loader, val_loader, _ = get_dataset(args.dataset)

    # calibration subset
    train_dataset = train_loader.dataset

    torch.manual_seed(CALIB_SEED)
    indices = torch.randperm(len(train_dataset))[:CALIB_SIZE]

    calib_dataset = Subset(train_dataset, indices)

    calib_loader = DataLoader(calib_dataset, batch_size=16, shuffle=False)

    print(f"Calibration samples: {len(calib_dataset)}")

    print("Loading model...")
    model = get_model(args.model, num_classes=10)

    print("Loading checkpoint...")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    model = model.to(device)
    model.eval()

    # FP32 baseline
    print("\nEvaluating FP32 model...")
    fp32_acc = evaluate(model, val_loader, device)
    print(f"FP32 Accuracy: {fp32_acc*100:.2f}%")

    # AdaRound
    print("\nApplying AdaRound...")
    # Step 1: AdaRound (optimize weights)
    model = apply_adaround(model, calib_loader, device)

    # Step 2: FX quantization (real INT8)
    quant_model = fx_quantize_model(model, calib_loader, device)

    # Step 3: Evaluate INT8 model
    acc = evaluate(
        quant_model,
        val_loader,
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    )

    print(f"\nAdaRound Accuracy: {acc*100:.2f}%")

    # Save
    save_dir = f"results/Phase3_Results/checkpoints/{args.model.upper()}"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{args.model}_{args.dataset}_adaround_int8.pth")

    torch.save(quant_model.state_dict(), save_path)

    print(f"Saved model to: {save_path}")

    # Log
    log_file = "results/Phase3_Results/logs/adaround_results.csv"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "a") as f:
        f.write(f"{args.dataset},{args.model},AdaRound,{acc*100:.2f}\n")


if __name__ == "__main__":
    main()
