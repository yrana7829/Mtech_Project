import os
import random
import shutil

RAW_BASE = "datasets/raw"
DEST_BASE = "datasets/standardized"

TRAIN = 500
VAL = 100
TEST = 100
TOTAL = TRAIN + VAL + TEST


EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

NWPU_CLASSES = [
    "airplane",
    "beach",
    "forest",
    "harbor",
    "intersection",
    "parking_lot",
    "river",
    "storage_tank",
    "tennis_court",
    "island",
]


def copy_images(files, dst):

    os.makedirs(dst, exist_ok=True)

    for src in files:

        filename = os.path.basename(src)

        shutil.copy(src, os.path.join(dst, filename))


def process_flat_dataset(dataset_name, classes):

    for cls in classes:

        src_dir = os.path.join(RAW_BASE, dataset_name, cls)

        images = os.listdir(src_dir)

        random.shuffle(images)

        selected = images[:TOTAL]

        train = selected[:TRAIN]
        val = selected[TRAIN : TRAIN + VAL]
        test = selected[TRAIN + VAL : TRAIN + VAL + TEST]

        train_paths = [os.path.join(src_dir, f) for f in train]
        val_paths = [os.path.join(src_dir, f) for f in val]
        test_paths = [os.path.join(src_dir, f) for f in test]

        copy_images(train_paths, f"{DEST_BASE}/{dataset_name}/train/{cls}")
        copy_images(val_paths, f"{DEST_BASE}/{dataset_name}/val/{cls}")
        copy_images(test_paths, f"{DEST_BASE}/{dataset_name}/test/{cls}")

        print(dataset_name, cls, "done")


def process_imagenet():

    dataset_name = "imagenet10"

    classes = os.listdir(os.path.join(RAW_BASE, dataset_name))

    process_flat_dataset(dataset_name, classes)


def process_nwpu():

    for cls in NWPU_CLASSES:

        train_dir = os.path.join(RAW_BASE, "nwpu_raw", "train", cls)
        test_dir = os.path.join(RAW_BASE, "nwpu_raw", "test", cls)

        train_imgs = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
        test_imgs = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]

        all_imgs = train_imgs + test_imgs

        random.shuffle(all_imgs)

        selected = all_imgs[:TOTAL]

        train = selected[:TRAIN]
        val = selected[TRAIN : TRAIN + VAL]
        test = selected[TRAIN + VAL : TRAIN + VAL + TEST]

        copy_images(train, f"{DEST_BASE}/nwpu10/train/{cls}")
        copy_images(val, f"{DEST_BASE}/nwpu10/val/{cls}")
        copy_images(test, f"{DEST_BASE}/nwpu10/test/{cls}")

        print("nwpu10", cls, "done")


def main():

    print("\nProcessing EuroSAT")
    process_flat_dataset("eurosat", EUROSAT_CLASSES)

    print("\nProcessing ImageNet10")
    process_imagenet()

    print("\nProcessing NWPU")
    process_nwpu()


if __name__ == "__main__":
    main()
