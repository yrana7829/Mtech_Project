import os
import random
import shutil
from pathlib import Path

SOURCE = "datasets/raw/eurosat"
DEST = "datasets/processed/eurosat"

train_ratio = 0.8
val_ratio = 0.1


def split_class(class_dir):

    images = os.listdir(class_dir)
    random.shuffle(images)

    n = len(images)

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    return (images[:train_end], images[train_end:val_end], images[val_end:])


def copy_images(files, src_dir, dst_dir):

    os.makedirs(dst_dir, exist_ok=True)

    for f in files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))


def prepare():

    for cls in os.listdir(SOURCE):
        class_path = os.path.join(SOURCE, cls)

        # Ignore anything that isn't a class directory (e.g. label_map.json)
        if not os.path.isdir(class_path):
            continue

        train, val, test = split_class(class_path)

        copy_images(train, class_path, f"{DEST}/train/{cls}")
        copy_images(val, class_path, f"{DEST}/val/{cls}")
        copy_images(test, class_path, f"{DEST}/test/{cls}")


if __name__ == "__main__":
    prepare()
