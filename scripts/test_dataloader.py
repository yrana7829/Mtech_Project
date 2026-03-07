import sys
import os

# add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.dataloader import get_dataset


def main():

    dataset_name = "nwpu10"

    train_loader, val_loader, test_loader = get_dataset(dataset_name)

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    images, labels = next(iter(train_loader))

    print("Batch image shape:", images.shape)
    print("Batch labels shape:", labels.shape)


if __name__ == "__main__":
    main()
