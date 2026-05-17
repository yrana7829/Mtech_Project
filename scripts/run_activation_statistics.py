import argparse
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset

from src.models.model_loader import get_model

from src.quantization.sensitivity.activation_statistics import (
    ActivationStatisticsCollector,
)

# ---------------------------------
# Config
# ---------------------------------
NUM_BATCHES = 50


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

    # ---------------------------------
    # Model
    # ---------------------------------
    print("Loading model...")

    model = get_model(args.model, num_classes=10)

    print("Loading checkpoint...")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    model.eval()

    # ---------------------------------
    # Target operators
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
    # Hook collector
    # ---------------------------------
    collector = ActivationStatisticsCollector(target_layers)

    collector.register_hooks(model)

    # ---------------------------------
    # Forward pass
    # ---------------------------------
    print("\nCollecting activations...")

    with torch.no_grad():

        for batch_idx, (images, _) in enumerate(train_loader):

            model(images.cpu())

            if batch_idx >= NUM_BATCHES:
                break

    collector.remove_hooks()

    # ---------------------------------
    # Statistics
    # ---------------------------------
    print("\nComputing statistics...")

    df = collector.compute_statistics()

    print("\nResults:\n")
    print(df)

    # ---------------------------------
    # Save
    # ---------------------------------
    save_dir = "results/" "Phase4_ActivationSensitivity"

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir, f"{args.model}_" f"{args.dataset}_" f"activation_statistics.csv"
    )

    df.to_csv(save_path, index=False)

    print(f"\n[INFO] Results saved:\n" f"{save_path}")


if __name__ == "__main__":
    main()
