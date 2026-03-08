import sys
import os
import torch
import argparse

from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

torch.backends.quantized.engine = "fbgemm"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate


def ptq_fx(model, calibration_loader):

    device = torch.device("cpu")

    model.eval()
    model.to(device)

    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}

    # Trace using a real sample from your dataset
    example_inputs = next(iter(calibration_loader))[0][:1].to(device)

    print("Preparing FX quantization...")
    prepared_model = prepare_fx(model, qconfig_dict, example_inputs)

    print("Running calibration...")
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            images = images.to(device)
            prepared_model(images)

            if i > 50:  # calibration batches
                break

    print("Converting to INT8...")
    quantized_model = convert_fx(prepared_model)

    return quantized_model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    print("Loading dataset...")
    train_loader, _, test_loader = get_dataset(args.dataset)

    print("Loading model...")
    model = get_model(args.model, num_classes=10)

    print("Loading FP32 checkpoint...")
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    quant_model = ptq_fx(model, train_loader)

    print("\nEvaluating quantized model...\n")
    acc = evaluate(quant_model, test_loader, torch.device("cpu"))

    print(f"\nFX PTQ Accuracy: {acc*100:.2f}%")

    os.makedirs("results/ptq_results", exist_ok=True)
    with open("results/ptq_results/fx_ptq_results.txt", "a") as f:
        f.write(f"{args.dataset},{args.model},{acc*100:.2f}\n")


if __name__ == "__main__":
    main()
