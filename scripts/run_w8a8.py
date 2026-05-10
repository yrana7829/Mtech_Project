import argparse
import os
import sys
import torch
import numpy as np
import random

from torch.utils.data import Subset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate

from src.quantization.ptq.full_ptq import apply_full_ptq


CALIB_SIZE = 1000
CALIB_SEED = 42

torch.manual_seed(CALIB_SEED)
np.random.seed(CALIB_SEED)
random.seed(CALIB_SEED)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    device = torch.device("cpu")

    # -----------------------------
    # Dataset
    # -----------------------------
    print("Loading dataset...")

    train_loader, val_loader, _ = get_dataset(args.dataset)

    train_dataset = train_loader.dataset

    indices = torch.randperm(len(train_dataset))[:CALIB_SIZE]

    calib_dataset = Subset(train_dataset, indices)

    calib_loader = DataLoader(calib_dataset, batch_size=16, shuffle=False)

    print(f"Calibration samples: {len(calib_dataset)}")

    # -----------------------------
    # Model
    # -----------------------------
    print("Loading model...")

    model = get_model(args.model, num_classes=10)

    print("Loading checkpoint...")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    model.eval()

    # -----------------------------
    # FP32 baseline
    # -----------------------------
    print("\nEvaluating FP32 model...")

    fp32_acc = evaluate(model, val_loader, device)

    print(f"FP32 Accuracy: {fp32_acc * 100:.2f}%")

    # -----------------------------
    # W8A8 PTQ
    # -----------------------------
    print("\nApplying W8A8 PTQ...")

    quantized_model = apply_full_ptq(model, calib_loader)

    # -----------------------------
    # Evaluate
    # -----------------------------
    print("\nEvaluating W8A8 model...")

    quant_acc = evaluate(quantized_model, val_loader, device)

    print(f"W8A8 Accuracy: {quant_acc * 100:.2f}%")

    # -----------------------------
    # Save results
    # -----------------------------
    save_dir = f"results/Phase4_Decomposition/{args.model.upper()}"

    os.makedirs(save_dir, exist_ok=True)

    result_file = os.path.join(save_dir, f"{args.model}_{args.dataset}_w8a8.txt")

    with open(result_file, "w") as f:

        f.write(f"FP32,{fp32_acc * 100:.2f}\n")
        f.write(f"W8A8,{quant_acc * 100:.2f}\n")

    print(f"\n[INFO] Results saved to: {result_file}")


if __name__ == "__main__":
    main()
