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

from src.quantization.sensitivity.static_activation_sensitivity import (
    collect_activation_ranges,
)

from src.quantization.sensitivity.activation_reconstruction import (
    ActivationReconstructionAnalyzer,
)

# ---------------------------------
# Config
# ---------------------------------
CALIB_SIZE = 1000
NUM_EVAL_SAMPLES = 200
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ---------------------------------
# Main
# ---------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)

    parser.add_argument("--model", required=True)

    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    device = torch.device("cpu")

    # ---------------------------------
    # Dataset
    # ---------------------------------
    print("Loading dataset...")

    train_loader, _, _ = get_dataset(args.dataset)

    train_dataset = train_loader.dataset

    # ---------------------------------
    # Calibration subset
    # ---------------------------------
    calib_indices = torch.randperm(len(train_dataset))[:CALIB_SIZE]

    calib_dataset = Subset(train_dataset, calib_indices)

    calib_loader = DataLoader(
        calib_dataset, batch_size=16, shuffle=False, num_workers=2
    )

    # ---------------------------------
    # Evaluation subset
    # ---------------------------------
    eval_indices = torch.randperm(len(train_dataset))[:NUM_EVAL_SAMPLES]

    eval_dataset = Subset(train_dataset, eval_indices)

    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=2)

    # ---------------------------------
    # Model
    # ---------------------------------
    print("Loading model...")

    model = get_model(args.model, num_classes=10)

    print("Loading checkpoint...")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    model.eval()

    # ---------------------------------
    # Target layers
    # ---------------------------------
    target_layers = [
        # Sensitive
        "features.0.0",
        "features.1.conv.0.0",
        "features.2.conv.1.0",
        # Stable
        "features.8.conv.1.0",
        "features.12.conv.1.0",
        "features.16.conv.1.0",
    ]

    print("\nTarget layers:")

    for layer in target_layers:
        print(layer)

    # ---------------------------------
    # Calibration ranges
    # ---------------------------------
    print("\nCollecting activation ranges...")

    activation_ranges = collect_activation_ranges(model, calib_loader)

    # ---------------------------------
    # Reconstruction analyzer
    # ---------------------------------
    analyzer = ActivationReconstructionAnalyzer(target_layers, activation_ranges)

    analyzer.register_hooks(model)

    # ---------------------------------
    # Run inference
    # ---------------------------------
    print("\nRunning reconstruction analysis...")

    with torch.no_grad():

        for images, _ in eval_loader:

            model(images.cpu())

    analyzer.remove_hooks()

    # ---------------------------------
    # Results
    # ---------------------------------
    df = analyzer.compute_results()

    print("\nResults:\n")

    print(df)

    # ---------------------------------
    # Save
    # ---------------------------------
    save_dir = "results/" "Phase4_ActivationSensitivity"

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir, f"{args.model}_" f"{args.dataset}_" f"activation_reconstruction.csv"
    )

    df.to_csv(save_path, index=False)

    print(f"\n[INFO] Results saved:\n" f"{save_path}")


if __name__ == "__main__":
    main()
