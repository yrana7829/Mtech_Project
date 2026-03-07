import os
import random
import shutil

DATASETS = ["eurosat", "imagenet10", "nwpu10"]

SOURCE_BASE = "datasets/processed"
DEST_BASE = "datasets/standardized"

TARGET_PER_CLASS = 700

TRAIN = 500
VAL = 100
TEST = 100


def copy_images(file_paths, dst_dir):

    os.makedirs(dst_dir, exist_ok=True)

    for src_path in file_paths:
        filename = os.path.basename(src_path)
        shutil.copy(src_path, os.path.join(dst_dir, filename))


def process_dataset(dataset):

    class_dir = os.path.join(SOURCE_BASE, dataset, "train")
    classes = os.listdir(class_dir)

    for cls in classes:

        all_images = []

        # collect images from train/val/test
        for split in ["train", "val", "test"]:

            path = os.path.join(SOURCE_BASE, dataset, split, cls)

            if os.path.exists(path):

                for img in os.listdir(path):
                    full_path = os.path.join(path, img)
                    all_images.append(full_path)

        random.shuffle(all_images)

        selected = all_images[:TARGET_PER_CLASS]

        train_imgs = selected[:TRAIN]
        val_imgs = selected[TRAIN : TRAIN + VAL]
        test_imgs = selected[TRAIN + VAL : TRAIN + VAL + TEST]

        copy_images(train_imgs, os.path.join(DEST_BASE, dataset, "train", cls))

        copy_images(val_imgs, os.path.join(DEST_BASE, dataset, "val", cls))

        copy_images(test_imgs, os.path.join(DEST_BASE, dataset, "test", cls))

        print(dataset, cls, "done")


def main():

    for dataset in DATASETS:
        print("\nProcessing:", dataset)
        process_dataset(dataset)


if __name__ == "__main__":
    main()
