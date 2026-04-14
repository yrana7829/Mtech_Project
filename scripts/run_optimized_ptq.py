import sys
import os
import torch
import argparse
import numpy as np
import random
from torch.utils.data import Subset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate
from src.quantization.ptq.optimized_ptq import optimized_ptq_fx


# -------------------------------
# Global config
# -------------------------------
CALIB_SIZE = 1000
CALIB_SEED = 42

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    print("Loading dataset...")
    train_loader, val_loader, _ = get_dataset(args.dataset)

    # -------------------------------
    # FIXED CALIBRATION LOADER
    # -------------------------------
    train_dataset = train_loader.dataset

    torch.manual_seed(CALIB_SEED)

    indices = torch.randperm(len(train_dataset))[:CALIB_SIZE]

    calib_dataset = Subset(train_dataset, indices)

    calib_loader = DataLoader(
        calib_dataset, batch_size=train_loader.batch_size, shuffle=False, num_workers=0
    )

    print(f"Calibration samples: {len(calib_dataset)}")

    print("Loading model...")
    model = get_model(args.model, num_classes=10)

    print("Loading FP32 checkpoint...")
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    # -------------------------------
    # RUN OPTIMIZED PTQ (FX)
    # -------------------------------
    quant_model = optimized_ptq_fx(model, calib_loader)

    print("\nEvaluating optimized PTQ model...\n")

    acc = evaluate(quant_model, val_loader, torch.device("cpu"))

    print(f"\nFX Optimized PTQ Accuracy: {acc*100:.2f}%")

    # -------------------------------
    # SAVE MODEL
    # -------------------------------
    model_dir = "results/Phase3_Results/checkpoints/MNV2"
    os.makedirs(model_dir, exist_ok=True)

    save_path = os.path.join(model_dir, "mnv2_eurosat_fx_optimized_ptq_int8.pth")

    torch.save(quant_model.state_dict(), save_path)

    print(f"Saved model to: {save_path}")

    # -------------------------------
    # SAVE LOG
    # -------------------------------
    log_dir = "results/Phase3_Results/logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "fx_optimized_ptq_mnv2_eurosat_int8.csv")

    with open(log_file, "a") as f:
        f.write(f"{args.dataset},{args.model},FX_Optimized,{acc*100:.2f}\n")


if __name__ == "__main__":
    main()
