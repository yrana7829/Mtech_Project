import sys
import os
import torch
import argparse
import numpy as np
import random
from torch.utils.data import Subset, DataLoader

# Ensure project root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate
from src.evaluation.performance import get_model_size, measure_latency

from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx


# -------------------------------
# Global config
# -------------------------------
CALIB_SIZE = 1000
CALIB_SEED = 42

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

torch.backends.quantized.engine = "fbgemm"


# -------------------------------
# Naive CLE (NOTE: simplified, not true CLE)
# -------------------------------
def cross_layer_equalization(model):

    print("Applying Naive CLE (independent channel scaling)...")

    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):

            w = module.weight.data

            w_view = w.view(w.size(0), -1)

            max_per_channel = w_view.abs().max(dim=1)[0]

            scale = max_per_channel.mean() / (max_per_channel + 1e-8)
            scale = scale.view(-1, 1, 1, 1)

            module.weight.data *= scale

    return model


# -------------------------------
# CLE + FX PTQ
# -------------------------------
def cle_ptq_fx(model, calibration_loader):

    device = torch.device("cpu")

    model.eval()
    model.to(device)

    # Apply naive CLE
    model = cross_layer_equalization(model)

    # QConfig (modern API)
    qconfig = get_default_qconfig("fbgemm")
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    example_inputs = next(iter(calibration_loader))[0][:1].to(device)

    print("Preparing FX quantization...")
    prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)

    print("Running calibration...")
    with torch.no_grad():
        for images, _ in calibration_loader:
            images = images.to(device)
            prepared_model(images)

    print("Converting to INT8...")
    quantized_model = convert_fx(prepared_model)

    return quantized_model


# -------------------------------
# Main
# -------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    # -------------------------------
    # Load dataset
    # -------------------------------
    print("Loading dataset...")
    train_loader, val_loader, _ = get_dataset(args.dataset)

    # -------------------------------
    # Calibration subset
    # -------------------------------
    train_dataset = train_loader.dataset

    torch.manual_seed(CALIB_SEED)
    indices = torch.randperm(len(train_dataset))[:CALIB_SIZE]

    calib_dataset = Subset(train_dataset, indices)

    calib_loader = DataLoader(
        calib_dataset, batch_size=train_loader.batch_size, shuffle=False, num_workers=0
    )

    print(f"Calibration samples: {len(calib_dataset)}")

    # -------------------------------
    # Load model
    # -------------------------------
    print("Loading model...")
    model = get_model(args.model, num_classes=10)

    print("Loading FP32 checkpoint...")
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    # -------------------------------
    # Run CLE PTQ
    # -------------------------------
    quant_model = cle_ptq_fx(model, calib_loader)

    # -------------------------------
    # Evaluation
    # -------------------------------
    print("\nEvaluating model...\n")

    acc = evaluate(quant_model, val_loader, torch.device("cpu"))

    print(f"CLE PTQ Accuracy: {acc*100:.2f}%")

    # -------------------------------
    # Save model (FIXED)
    # -------------------------------
    model_dir = f"results/Phase3_Results/checkpoints/{args.model.upper()}"
    os.makedirs(model_dir, exist_ok=True)

    save_path = os.path.join(
        model_dir, f"{args.model}_{args.dataset}_fx_cle_ptq_int8.pth"
    )

    torch.save(quant_model.state_dict(), save_path)

    print(f"Saved model to: {save_path}")

    # -------------------------------
    # Metrics
    # -------------------------------
    model_size = get_model_size(save_path)
    latency = measure_latency(quant_model, val_loader, torch.device("cpu"))

    print(f"Model Size: {model_size:.2f} MB")
    print(f"Latency: {latency:.2f} ms")

    # -------------------------------
    # Logging (FIXED)
    # -------------------------------
    log_dir = "results/Phase3_Results/logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "fx_ptq_results.csv")

    with open(log_file, "a") as f:
        f.write(
            f"{args.dataset},{args.model},CLE,"
            f"{acc*100:.2f},{model_size:.2f},{latency:.2f}\n"
        )


if __name__ == "__main__":
    main()
