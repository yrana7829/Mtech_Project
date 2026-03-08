import sys
import os
import torch
import argparse
import torch.nn as nn
import torch.quantization as quant

# quantization backend
torch.backends.quantized.engine = "fbgemm"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate


# ---------------------------------------------------
# Quantization Wrapper
# ---------------------------------------------------
class QuantizedModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.quant = quant.QuantStub()
        self.model = model
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


# ---------------------------------------------------
# Naive PTQ Function
# ---------------------------------------------------
def naive_ptq(model, calibration_loader):

    device = torch.device("cpu")

    model.eval()
    model.to(device)

    torch.backends.quantized.engine = "fbgemm"

    # attach qconfig
    model.qconfig = quant.get_default_qconfig("fbgemm")

    print("Preparing model for calibration...")
    quant.prepare(model, inplace=True)

    print("Running calibration...")

    with torch.no_grad():

        for i, (images, _) in enumerate(calibration_loader):

            images = images.to(device)

            model(images)

            if i > 50:
                break

    print("Converting to INT8...")

    quantized_model = quant.convert(model, inplace=False)

    return quantized_model


# ---------------------------------------------------
# Main
# ---------------------------------------------------
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

    # IMPORTANT FIX: wrap model
    model = QuantizedModel(model)

    # run PTQ
    quant_model = naive_ptq(model, train_loader)

    print("\nEvaluating quantized model...\n")

    acc = evaluate(quant_model, test_loader, torch.device("cpu"))

    print(f"\nNaive PTQ Accuracy: {acc*100:.2f}%")

    os.makedirs("results/ptq_results", exist_ok=True)

    with open("results/ptq_results/naive_ptq_results.txt", "a") as f:
        f.write(f"{args.dataset},{args.model},{acc*100:.2f}\n")


if __name__ == "__main__":
    main()
