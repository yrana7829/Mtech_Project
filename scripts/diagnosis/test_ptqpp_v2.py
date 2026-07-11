"""
scripts/diagnosis/test_ptqpp_v2.py

NOTE:
This is a ready-to-integrate scaffold for the PTQ++ v2 diagnosis script.
Because the original project-specific implementations (dataset loader,
evaluation function, model loader, and PTQ++ pipeline) are not available
inside this execution environment, you should paste your existing
build_fx_quantized_model() implementation where indicated if it differs.

The imports and workflow match the requested project structure.
"""

import copy
import random
import numpy as np
import torch

from torch.utils.data import DataLoader, Subset
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate
from src.quantization.proposed.proposed_ptq_pipeline_v2 import (
    apply_proposed_ptq_pipeline_v2,
)

SEED = 42
CALIBRATION_SAMPLES = 1000


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_fx_quantized_model(model, calibration_loader):
    model.eval()

    qconfig = get_default_qconfig("fbgemm")
    qconfig_mapping = (
        torch.ao.quantization.QConfigMapping()
        .set_global(qconfig)
    )

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

    train_dataset, test_dataset = get_dataset()

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(train_dataset))[:CALIBRATION_SAMPLES]

    calib_loader = DataLoader(
        Subset(train_dataset, indices),
        batch_size=32,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
    )

    model = get_model()
    model.eval()

    fp32_acc = evaluate(model, test_loader)

    ptqpp_model, allocation = apply_proposed_ptq_pipeline_v2(model)

    ptqpp_fp32_acc = evaluate(ptqpp_model, test_loader)

    int8_model = build_fx_quantized_model(
        ptqpp_model,
        calib_loader,
    )

    int8_acc = evaluate(int8_model, test_loader)

    print("\n==============================")
    print("Layer Allocation Summary")
    print("==============================")

    if allocation is None:
        print("No allocation information returned.")
    else:
        int8 = allocation.get("int8", [])
        int6 = allocation.get("int6", [])

        print("\nINT8 Layers")
        for x in int8:
            print("  ", x)

        print("\nINT6 Recommended Layers")
        for x in int6:
            print("  ", x)

    preprocess_loss = fp32_acc - ptqpp_fp32_acc
    quant_loss = ptqpp_fp32_acc - int8_acc
    total_loss = fp32_acc - int8_acc

    print("\n==============================")
    print("Final Comparison")
    print("==============================")
    print(f"{'FP32 Accuracy':35s}: {fp32_acc:.2f}%")
    print(f"{'PTQ++ FP32 Accuracy':35s}: {ptqpp_fp32_acc:.2f}%")
    print(f"{'PTQ++ INT8 Accuracy':35s}: {int8_acc:.2f}%")
    print(f"{'Accuracy loss after preprocessing':35s}: {preprocess_loss:.2f}%")
    print(f"{'Accuracy loss after FX quantization':35s}: {quant_loss:.2f}%")
    print(f"{'Total loss from FP32':35s}: {total_loss:.2f}%")


if __name__ == "__main__":
    main()
