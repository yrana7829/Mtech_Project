import os
import random
import shutil

SOURCE = "datasets/raw/nwpu_resisc45/train"
DEST = "datasets/processed/nwpu10"

CLASSES = [
    "airplane",
    "beach",
    "forest",
    "harbor",
    "intersection",
    "parking_lot",
    "river",
    "storage_tank",
    "tennis_court",
    "residential",
]

train_ratio = 0.8
val_ratio = 0.1


def split_images(images):

    random.shuffle(images)

    n = len(images)

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]

    return train, val, test


def copy_images(files, src_dir, dst_dir):

    os.makedirs(dst_dir, exist_ok=True)

    for f in files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))


def prepare_dataset():

    for cls in CLASSES:

        class_path = os.path.join(SOURCE, cls)

        if not os.path.exists(class_path):
            print(f"Class {cls} not found")
            continue

        images = os.listdir(class_path)

        train, val, test = split_images(images)

        copy_images(train, class_path, f"{DEST}/train/{cls}")
        copy_images(val, class_path, f"{DEST}/val/{cls}")
        copy_images(test, class_path, f"{DEST}/test/{cls}")

        print(f"{cls} done")


if __name__ == "__main__":
    prepare_dataset()
