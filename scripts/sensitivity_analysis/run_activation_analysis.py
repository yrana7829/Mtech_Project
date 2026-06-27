import sys
import os
import torch
import argparse
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.analysis.activation_stats import ActivationStatsCollector


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    print("Loading dataset...")
    train_loader, val_loader, _ = get_dataset(args.dataset)

    print("Loading model...")
    model = get_model(args.model, num_classes=10)

    print("Loading checkpoint...")
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    model.eval()
    model.to("cpu")

    # -------------------------------
    # Register hooks
    # -------------------------------
    collector = ActivationStatsCollector()
    collector.register_hooks(model)

    print("Collecting activation statistics...")

    with torch.no_grad():

        for i, (images, _) in enumerate(val_loader):

            images = images.to("cpu")
            model(images)

            if i >= 20:  # limit batches
                break

    collector.remove_hooks()

    stats = collector.aggregate()

    # -------------------------------
    # Save results
    # -------------------------------
    os.makedirs("results/Phase3_Results/analysis", exist_ok=True)

    save_path = os.path.join(
        "results/Phase3_Results/analysis",
        f"{args.model}_{args.dataset}_activation_stats.json",
    )

    with open(save_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Saved activation stats to: {save_path}")


if __name__ == "__main__":
    main()
