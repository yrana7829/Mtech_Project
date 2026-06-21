import os
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(project_root)

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate

# CHECKPOINT_PATH = "results/checkpoints/mobilenet_v2_eurosat_rgb_fp32.pth"
CHECKPOINT_PATH = "results/checkpoints/eurosat_mobilenetv2_fp32.pth"
DATASET_NAME = "eurosat"
NUM_CLASSES = 10


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")

    _, _, test_loader = get_dataset(DATASET_NAME)

    print("Loading model...")

    model = get_model("mobilenetv2", NUM_CLASSES)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(checkpoint)

    model.to(device)

    print("Evaluating...")

    acc = evaluate(model, test_loader, device)

    print(f"\nFP32 Checkpoint Accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
