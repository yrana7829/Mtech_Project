import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
import platform

# ---------------------------------------------------------
# ImageNet normalization used in the original test pipeline
# ---------------------------------------------------------

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)

STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def preprocess_image(image_path):
    """
    Reproduce the deployment test preprocessing:

    1. Open image
    2. Convert to RGB
    3. Resize to 224 x 224
    4. Convert to float32 in [0, 1]
    5. HWC -> CHW
    6. ImageNet normalization
    7. Add batch dimension
    """

    image = Image.open(image_path).convert("RGB")

    image = image.resize((224, 224), resample=Image.Resampling.BILINEAR)

    image = np.asarray(image, dtype=np.float32) / 255.0

    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    # Normalize
    image = (image - MEAN) / STD

    # Add batch dimension: [3,224,224] -> [1,3,224,224]
    image = np.expand_dims(image, axis=0)

    return np.ascontiguousarray(image, dtype=np.float32)


def get_dataset_samples(data_dir):
    """
    Reproduce torchvision ImageFolder class ordering:
    class folders are sorted alphabetically.
    """

    data_dir = Path(data_dir)

    class_names = sorted(
        [folder.name for folder in data_dir.iterdir() if folder.is_dir()]
    )

    class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}

    samples = []

    for class_name in class_names:

        class_dir = data_dir / class_name
        label = class_to_idx[class_name]

        for image_path in sorted(class_dir.rglob("*")):

            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, label))

    return class_names, class_to_idx, samples


def main(args):

    print("\n==========================================")
    print(" Raspberry Pi ONNX Evaluation")
    print("==========================================\n")

    print(f"Host              : {platform.node()}")
    print(f"Architecture      : {platform.machine()}")
    print(f"Python            : {platform.python_version()}")
    print(f"ONNX Runtime      : {ort.__version__}")

    try:
        model = subprocess.check_output(
            ["cat", "/proc/device-tree/model"], text=True
        ).strip("\x00")
        print(f"Hardware          : {model}")
    except Exception:
        pass

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    class_names, class_to_idx, samples = get_dataset_samples(args.data_dir)

    print("Classes:")
    for class_name, index in class_to_idx.items():
        print(f"  {index}: {class_name}")

    print(f"\nTotal test images: {len(samples)}")

    if len(samples) == 0:
        raise RuntimeError("No test images found.")

    # -----------------------------------------------------
    # ONNX Runtime session
    # -----------------------------------------------------

    print(f"\nLoading model:")
    print(args.model)

    session_options = ort.SessionOptions()

    session = ort.InferenceSession(
        args.model, sess_options=session_options, providers=["CPUExecutionProvider"]
    )

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    input_name = input_info.name

    print("\nONNX Runtime version:", ort.__version__)
    print("Execution providers:", session.get_providers())
    print("Input name:", input_name)
    print("Input shape:", input_info.shape)
    print("Output shape:", output_info.shape)

    # -----------------------------------------------------
    # Warm-up
    # -----------------------------------------------------

    print(f"\nRunning {args.warmup} warm-up inferences...")

    warmup_input = preprocess_image(samples[0][0])

    for _ in range(args.warmup):
        session.run(None, {input_name: warmup_input})

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    print("\nRunning evaluation...\n")

    correct = 0
    total = 0

    inference_times = []

    start_total = time.perf_counter()

    for index, (image_path, label) in enumerate(samples, start=1):

        input_tensor = preprocess_image(image_path)

        start = time.perf_counter()

        output = session.run(None, {input_name: input_tensor})[0]

        elapsed = time.perf_counter() - start

        inference_times.append(elapsed)

        prediction = int(np.argmax(output, axis=1)[0])

        if prediction == label:
            correct += 1

        total += 1

        if index % 100 == 0:
            running_accuracy = 100.0 * correct / total

            print(
                f"Processed {index:4d}/{len(samples)}"
                f" | Accuracy: {running_accuracy:.2f}%"
            )

    total_elapsed = time.perf_counter() - start_total

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    times_ms = np.array(inference_times) * 1000.0

    accuracy = 100.0 * correct / total

    model_size_mb = os.path.getsize(args.model) / (1024 * 1024)

    print("\n==========================================")
    print(" RESULTS")
    print("==========================================")

    print(f"ONNX Runtime : {ort.__version__}")
    print(f"CPU Provider : {session.get_providers()[0]}")
    print(f"Images/sec   : {total / total_elapsed:.2f}")

    print(f"Total images          : {total}")
    print(f"Correct predictions   : {correct}")
    print(f"Accuracy              : {accuracy:.2f}%")

    print()

    print(f"Mean inference latency: {np.mean(times_ms):.3f} ms")
    print(f"Median latency        : {np.median(times_ms):.3f} ms")
    print(f"Minimum latency       : {np.min(times_ms):.3f} ms")
    print(f"Maximum latency       : {np.max(times_ms):.3f} ms")
    print(f"P95 latency           : {np.percentile(times_ms, 95):.3f} ms")

    print()

    print(f"Total evaluation time : {total_elapsed:.2f} s")
    print(f"Model file size       : {model_size_mb:.2f} MB")

    print("==========================================\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, help="Path to ONNX model")

    parser.add_argument(
        "--data_dir", required=True, help="Path to ImageFolder-style test dataset"
    )

    parser.add_argument(
        "--warmup", type=int, default=20, help="Number of warm-up inferences"
    )

    args = parser.parse_args()

    main(args)
