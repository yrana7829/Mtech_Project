import sys
import os
import torch
import argparse
import torch.quantization as quant

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model


def naive_ptq(model, calibration_loader):

    print("\nStarting Naive PTQ")

    device = torch.device("cpu")
    model.eval()
    model.to(device)

    torch.backends.quantized.engine = "fbgemm"

    # Fuse layers (required for ResNet PTQ)
    if hasattr(model, "fuse_model"):
        model.fuse_model()

    # Assign quantization config
    model.qconfig = quant.get_default_qconfig("fbgemm")

    print("Preparing model for calibration...")
    quant.prepare(model, inplace=True)

    # Calibration
    print("Running calibration...")
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            images = images.to(device)
            model(images)

            if i > 50:  # limit calibration batches
                break

    print("Converting model to INT8...")
    quantized_model = quant.convert(model, inplace=False)

    print("PTQ conversion finished\n")

    return quantized_model


def main():

    print("Script started")

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    device = torch.device("cpu")

    print("Loading dataset...")
    train_loader, _, _ = get_dataset(args.dataset)

    print("Loading model...")
    model = get_model(args.model, num_classes=10)

    print("Loading FP32 checkpoint...")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    quant_model = naive_ptq(model, train_loader)

    save_path = f"results/checkpoints/{args.dataset}_{args.model}_ptq.pth"

    print("Saving quantized weights...")
    torch.save(quant_model.state_dict(), save_path)

    print(f"PTQ weights saved to: {save_path}")


if __name__ == "__main__":
    main()
