import copy
import random
import numpy as np
import torch
import os
import sys

from torch.utils.data import DataLoader, Subset
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.insert(0, PROJECT_ROOT)

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate
from src.quantization.proposed.proposed_ptq_pipeline_v3 import (
    apply_proposed_ptq_pipeline_v3,
)

SEED = 42
CALIBRATION_SAMPLES = 1000
device = torch.device("cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_fx_quantized_model(model, calibration_loader):
    model.eval()

    qconfig = get_default_qconfig("fbgemm")
    qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global(qconfig)

    example_inputs = (torch.randn(1, 3, 224, 224),)

    prepared = prepare_fx(
        copy.deepcopy(model),
        qconfig_mapping,
        example_inputs,
    )

    with torch.no_grad():
        for images, _ in calibration_loader:
            prepared(images)

    quantized = convert_fx(prepared)
    return quantized


def main():
    set_seed()

    print("Loading dataset...")

    train_loader, _, test_loader = get_dataset("eurosat")

    train_dataset = train_loader.dataset

    torch.manual_seed(SEED)

    indices = torch.randperm(len(train_dataset))[:CALIBRATION_SAMPLES]

    calib_dataset = Subset(train_dataset, indices)

    calib_loader = DataLoader(
        calib_dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = get_model(
        "mobilenetv2",
        num_classes=10,
    )
    checkpoint = "results/checkpoints/eurosat_mobilenetv2_fp32.pth"

    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
    )

    model.load_state_dict(state_dict)

    model.eval()

    fp32_acc = evaluate(model, test_loader, device)

    ptqpp_model = apply_proposed_ptq_pipeline_v3(model, device)

    ptqpp_fp32_acc = evaluate(ptqpp_model, test_loader, device)

    int8_model = build_fx_quantized_model(
        ptqpp_model,
        calib_loader,
    )

    int8_acc = evaluate(
        int8_model,
        test_loader,
        device,
    )

    preprocess_loss = fp32_acc - ptqpp_fp32_acc
    quant_loss = ptqpp_fp32_acc - int8_acc
    total_loss = fp32_acc - int8_acc

    print("\n==============================")
    print("Final Comparison")
    print("==============================")
    print(f"{'FP32 Accuracy':35s}: {fp32_acc*100:.2f}%")

    print(f"{'PTQ++ FP32 Accuracy':35s}: {ptqpp_fp32_acc*100:.2f}%")
    print(f"{'PTQ++ INT8 Accuracy':35s}: {int8_acc*100:.2f}%")
    print(f"{'Accuracy loss after preprocessing':35s}: {preprocess_loss*100:.2f} pp")
    print(f"{'Accuracy loss after FX quantization':35s}: {quant_loss*100:.2f} pp")
    print(f"{'Total loss from FP32':35s}: {total_loss*100:.2f} pp")


if __name__ == "__main__":
    main()
