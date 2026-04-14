import sys
import os
import torch
import argparse
import torch.nn as nn
import torch.quantization as quant
import numpy as np
import random
from torch.utils.data import Subset, DataLoader

# quantization backend
torch.backends.quantized.engine = "fbgemm"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate

# -------------------------------
# Global config
# -------------------------------
CALIB_SIZE = 1000
CALIB_SEED = 42

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ---------------------------------------------------
# Quantization Wrapper
# ---------------------------------------------------
class QuantizedModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.quant = quant.QuantStub()
        self.model = model
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


# ---------------------------------------------------
# Naive PTQ Function
# ---------------------------------------------------
def naive_ptq(model, calibration_loader):

    device = torch.device("cpu")

    model.eval()
    model.to(device)

    # attach qconfig (PURE naive PTQ)
    model.qconfig = quant.get_default_qconfig("fbgemm")

    print("Preparing model for calibration...")
    quant.prepare(model, inplace=True)

    print("Running calibration on fixed subset...")

    with torch.no_grad():
        for images, _ in calibration_loader:
            images = images.to(device)
            model(images)

    print("Converting to INT8...")

    quantized_model = quant.convert(model, inplace=False)

    return quantized_model


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_dataset(args.dataset)

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

    # wrap model
    model = QuantizedModel(model)

    # run PTQ
    quant_model = naive_ptq(model, calib_loader)

    print("\nEvaluating quantized model...\n")

    model_dir = "results/Phase3_Results/checkpoints/MNV2"
    os.makedirs(model_dir, exist_ok=True)

    save_path = os.path.join(model_dir, "mnv2_eurosat_naive_ptq_int8.pth")

    torch.save(quant_model.state_dict(), save_path)

    print(f"Saved quantized model to: {save_path}")

    # 🔴 FIXED: use validation set
    acc = evaluate(quant_model, val_loader, torch.device("cpu"))

    print(f"\nNaive PTQ Accuracy: {acc*100:.2f}%")

    log_dir = "results/Phase3_Results/logs"

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "naive_ptq_mnv2_eurosat_int8.csv")

    with open(log_file, "a") as f:
        f.write(f"{args.dataset},{args.model},Naive,{acc*100:.2f}\n")


if __name__ == "__main__":
    main()
