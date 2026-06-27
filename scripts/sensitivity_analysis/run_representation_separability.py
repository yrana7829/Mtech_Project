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

from src.quantization.sensitivity.representation_separability import (
    RepresentationSeparabilityAnalyzer,
)

CALIB_SIZE = 1000
NUM_SAMPLES = 300
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)

    parser.add_argument("--model", required=True)

    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    device = torch.device("cpu")

    print("Loading dataset...")

    train_loader, _, _ = get_dataset(args.dataset)

    train_dataset = train_loader.dataset

    calib_indices = torch.randperm(len(train_dataset))[:CALIB_SIZE]

    calib_dataset = Subset(train_dataset, calib_indices)

    calib_loader = DataLoader(
        calib_dataset, batch_size=16, shuffle=False, num_workers=2
    )

    eval_indices = torch.randperm(len(train_dataset))[:NUM_SAMPLES]

    eval_dataset = Subset(train_dataset, eval_indices)

    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=2)

    print("Loading model...")

    model = get_model(args.model, num_classes=10)

    print("Loading checkpoint...")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    model.eval()

    target_layers = ["features.2", "features.16"]

    activation_ranges = collect_activation_ranges(model, calib_loader)

    analyzer = RepresentationSeparabilityAnalyzer(target_layers, activation_ranges)

    analyzer.register_hooks(model)

    # FP32
    print("\nRunning FP32...")

    analyzer.quant_mode = False

    with torch.no_grad():

        for images, labels in eval_loader:

            analyzer.labels.extend(labels.numpy().tolist())

            model(images)

    # Quantized
    print("\nRunning W32A8...")

    analyzer.quant_mode = True

    with torch.no_grad():

        for images, _ in eval_loader:

            model(images)

    analyzer.remove_hooks()

    df = analyzer.compute_metrics()

    print("\nResults:\n")

    print(df)

    save_dir = "results/" "Phase4_ActivationSensitivity"

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir, f"{args.model}_" f"{args.dataset}_" f"representation_separability.csv"
    )

    df.to_csv(save_path, index=False)

    print(f"\nSaved:\n" f"{save_path}")


if __name__ == "__main__":
    main()
