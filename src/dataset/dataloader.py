import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(img_size=224):

    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, test_transform


def get_dataset(dataset_name, batch_size=64):

    base_path = f"datasets/standardized/{dataset_name}"

    train_transform, test_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        os.path.join(base_path, "train"), transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(base_path, "val"), transform=test_transform
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(base_path, "test"), transform=test_transform
    )

    # 🔴 FIXED: shuffle=False and num_workers=0 for determinism
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader
