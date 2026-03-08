import sys
import os
import torch
import argparse
import torch.quantization as quant

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--quant_checkpoint", required=True)

    args = parser.parse_args()

    device = torch.device("cpu")

    print("Loading dataset...")
    _, _, test_loader = get_dataset(args.dataset)

    print("Rebuilding model architecture...")
    model = get_model(args.model, num_classes=10)

    model.eval()
    model.to(device)

    # same backend used during PTQ
    torch.backends.quantized.engine = "fbgemm"

    model.qconfig = quant.get_default_qconfig("fbgemm")

    print("Preparing quantized structure...")
    quant.prepare(model, inplace=True)
    quant.convert(model, inplace=True)

    print("Loading quantized weights...")
    model.load_state_dict(torch.load(args.quant_checkpoint, map_location="cpu"))

    model.eval()

    print("\nEvaluating quantized model...\n")

    acc = evaluate(model, test_loader, device)

    print(f"Quantized Accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
