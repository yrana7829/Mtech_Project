"""
scripts/export/export_ptqpp_v2_onnx.py

PTQ++ v2 Export Script (template)

This script is intentionally based on the working export_naive_ptq_onnx.py
pipeline. Replace the helper functions below with the exact implementations
from your validated exporter if they already exist in your project.

Pipeline
--------
FP32 checkpoint
    -> PTQ++ v2 preprocessing (TWC + Mixed Precision)
    -> FX PTQ (INT8)
    -> Evaluate
    -> Save INT8 checkpoint
    -> Export ONNX
    -> Validate ONNX
    -> Compare PyTorch vs ONNX
"""

import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import onnx
import onnxruntime as ort

from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate
from src.quantization.proposed.proposed_ptq_pipeline_v2 import (
    apply_proposed_ptq_pipeline_v2,
)

CALIB_SIZE = 1000
CALIB_SEED = 42


def set_all_seeds(seed=CALIB_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_fx_quantized_model(fp32_model, calibration_loader):
    device = torch.device("cpu")
    fp32_model.eval().to(device)

    torch.backends.quantized.engine = "fbgemm"
    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}

    example_inputs = next(iter(calibration_loader))[0][:1].to(device)

    prepared = prepare_fx(fp32_model, qconfig_dict, example_inputs)

    set_all_seeds(CALIB_SEED)

    with torch.no_grad():
        for images, _ in calibration_loader:
            prepared(images.to(device))

    quantized = convert_fx(prepared)
    quantized.eval()
    return quantized


def export_to_onnx(model, output_path):
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cpu")

    train_loader, _, test_loader = get_dataset(args.dataset)

    train_dataset = train_loader.dataset
    set_all_seeds()

    indices = torch.randperm(len(train_dataset))[:CALIB_SIZE]
    calib_loader = DataLoader(
        Subset(train_dataset, indices),
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = get_model(args.model, num_classes=10)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    fp32_acc = evaluate(model, test_loader, device)
    print(f"FP32 Accuracy: {fp32_acc*100:.2f}%")

    model = apply_proposed_ptq_pipeline_v2(model, device)

    ptqpp_fp32_acc = evaluate(model, test_loader, device)
    print(f"PTQ++ FP32 Accuracy: {ptqpp_fp32_acc*100:.2f}%")

    quant_model = build_fx_quantized_model(model, calib_loader)

    int8_acc = evaluate(quant_model, test_loader, device)
    print(f"PTQ++ INT8 Accuracy: {int8_acc*100:.2f}%")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    pth_path = os.path.splitext(args.output)[0] + ".pth"
    torch.save(quant_model.state_dict(), pth_path)
    print("Saved:", pth_path)

    export_to_onnx(quant_model, args.output)
    print("Exported:", args.output)

    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    print("Execution Providers:", sess.get_providers())

    print("\n==============================")
    print("PTQ++ v2 EXPORT SUMMARY")
    print("==============================")
    print(f"FP32 Accuracy      : {fp32_acc*100:.2f}%")
    print(f"PTQ++ FP32         : {ptqpp_fp32_acc*100:.2f}%")
    print(f"PTQ++ INT8         : {int8_acc*100:.2f}%")
    print(f"ONNX Valid         : YES")
    print(f"Checkpoint         : {pth_path}")
    print(f"ONNX               : {args.output}")


if __name__ == "__main__":
    main()
