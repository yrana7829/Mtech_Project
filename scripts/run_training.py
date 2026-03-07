import sys
import os
import torch
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.dataloader import get_dataset
from src.models.model_loader import get_model
from src.training.trainer import Trainer


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)

    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model
    epochs = args.epochs
    num_classes = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader = get_dataset(dataset_name)

    model = get_model(model_name, num_classes)

    trainer = Trainer(model, train_loader, val_loader, device)

    checkpoint_name = f"results/checkpoints/{dataset_name}_{model_name}_fp32.pth"

    trainer.train(epochs, save_path=checkpoint_name)


if __name__ == "__main__":
    main()
