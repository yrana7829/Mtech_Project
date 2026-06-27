import sys
import os
import torch
import argparse
import numpy as np
import random
from torch.utils.data import Subset, DataLoader

from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

torch.backends.quantized.engine = "fbgemm"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model

CALIB_SIZE = 1000
CALIB_SEED = 42

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def ptq_fx(model, calibration_loader):

    model.eval()

    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}

    example_inputs = next(iter(calibration_loader))[0][:1]

    prepared_model = prepare_fx(model, qconfig_dict, example_inputs)

    with torch.no_grad():

        for images, _ in calibration_loader:
            prepared_model(images)

    quantized_model = convert_fx(prepared_model)

    return quantized_model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)

    parser.add_argument("--model", required=True)

    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    print("Loading dataset...")

    train_loader, _, _ = get_dataset(args.dataset)

    train_dataset = train_loader.dataset

    torch.manual_seed(CALIB_SEED)

    indices = torch.randperm(len(train_dataset))[:CALIB_SIZE]

    calib_dataset = Subset(train_dataset, indices)

    calib_loader = DataLoader(
        calib_dataset, batch_size=train_loader.batch_size, shuffle=False
    )

    print("Loading model...")

    model = get_model(args.model, num_classes=10)

    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    print("Running PTQ...")

    quant_model = ptq_fx(model, calib_loader)

    print("\n========================")
    print("MODEL TYPE")
    print("========================")

    print(type(quant_model))

    print("\n========================")
    print("QUANTIZED MODULES")
    print("========================")

    count = 0

    for name, module in quant_model.named_modules():

        module_type = str(type(module))

        if "quantized" in module_type.lower() or "Quantized" in module_type:
            print(name)
            print(module_type)
            print()

            count += 1

    print(f"Total Quantized Modules: {count}")


if __name__ == "__main__":
    main()
