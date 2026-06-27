import sys
import os
import torch
import argparse
import json
import numpy as np
import random
from torch.utils.data import Subset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate
from src.evaluation.performance import get_model_size, measure_latency

from src.analysis.layer_selection import (
    compute_outlier_ratios,
    select_layers_for_clipping,
)

from src.quantization.ptq.layerwise_clipped_ptq import layerwise_clipped_ptq_fx

# -------------------------------
# Config
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
    parser.add_argument("--stats_path", required=True)
    parser.add_argument("--threshold", type=float, default=8.0)
    parser.add_argument("--percentile", type=float, default=99.0)

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

    print("Loading checkpoint...")
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    # -------------------------------
    # Load activation stats
    # -------------------------------
    print("Loading activation stats...")
    with open(args.stats_path, "r") as f:
        stats = json.load(f)

    # -------------------------------
    # Compute outlier ratios
    # -------------------------------
    ratios = compute_outlier_ratios(stats)

    # -------------------------------
    # Select layers
    # -------------------------------
    clipped_layers = select_layers_for_clipping(ratios, threshold=args.threshold)

    print(f"\nSelected {len(clipped_layers)} layers for clipping\n")

    # Debug print (important)
    for l in clipped_layers[:10]:
        print(f"CLIP: {l} | ratio={ratios[l]:.2f}")

    # -------------------------------
    # Run PTQ
    # -------------------------------
    quant_model = layerwise_clipped_ptq_fx(
        model, calib_loader, clipped_layers, percentile=args.percentile
    )

    # -------------------------------
    # Evaluation
    # -------------------------------
    print("\nEvaluating model...\n")

    acc = evaluate(quant_model, val_loader, torch.device("cpu"))

    print(f"Accuracy: {acc*100:.2f}%")

    # -------------------------------
    # Save model
    # -------------------------------
    model_dir = "results/Phase3_Results/checkpoints/MNV2"
    os.makedirs(model_dir, exist_ok=True)

    save_path = os.path.join(model_dir, "mnv2_eurosat_fx_layerwise_clipped_int8.pth")

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
    # Logging
    # -------------------------------
    log_dir = "results/Phase3_Results/logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "fx_layerwise_clipped_results.csv")

    with open(log_file, "a") as f:
        f.write(
            f"{args.dataset},{args.model},Layerwise_Clipped,"
            f"{acc*100:.2f},{model_size:.2f},{latency:.2f},"
            f"{len(clipped_layers)}\n"
        )


if __name__ == "__main__":
    main()
