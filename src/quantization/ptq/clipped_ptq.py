import torch
import torch.nn as nn

from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx


# -------------------------------
# Percentile clipping
# -------------------------------
def percentile_clip(tensor, percentile=99.9):

    flat = tensor.detach().view(-1)

    k = int(len(flat) * percentile / 100)

    k = max(1, min(k, len(flat)))  # safety

    threshold = flat.abs().kthvalue(k).values

    return torch.clamp(tensor, -threshold, threshold)


# -------------------------------
# Add hooks for activation clipping
# -------------------------------
def add_clipping_hooks(model, percentile=99.9):

    hooks = []

    def hook_fn(module, input, output):
        return percentile_clip(output, percentile)

    for name, module in model.named_modules():

        # Apply clipping after ReLU (important choice)
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn))

    return hooks


# -------------------------------
# FX PTQ with clipping
# -------------------------------
def clipped_ptq_fx(model, calibration_loader, percentile=99.9):

    device = torch.device("cpu")

    model.eval()
    model.to(device)

    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}

    # Example input for FX tracing
    example_inputs = next(iter(calibration_loader))[0][:1].to(device)

    print("Preparing FX quantization...")
    prepared_model = prepare_fx(model, qconfig_dict, example_inputs)

    # -------------------------------
    # Add clipping hooks
    # -------------------------------
    print(f"Applying activation clipping (p={percentile})...")
    hooks = add_clipping_hooks(prepared_model, percentile)

    # -------------------------------
    # Calibration
    # -------------------------------
    print("Running calibration...")
    with torch.no_grad():
        for images, _ in calibration_loader:
            images = images.to(device)
            prepared_model(images)

    # Remove hooks after calibration
    for h in hooks:
        h.remove()

    print("Converting to INT8...")
    quantized_model = convert_fx(prepared_model)

    return quantized_model
