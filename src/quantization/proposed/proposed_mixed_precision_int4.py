import torch
import torch.nn as nn


def quantize_weight(weight, num_bits):

    qmax = 2 ** (num_bits - 1) - 1

    # robust scale (prevents outliers from dominating)
    scale = torch.quantile(weight.abs(), 0.999) / qmax + 1e-8

    q = torch.round(weight / scale)
    q = torch.clamp(q, -qmax, qmax)

    return q * scale


def apply_proposed_mixed_precision_int4(model):

    print("\nApplying Proposed Mixed Precision INT4...\n")

    for name, module in model.named_modules():

        # CASE 1: LPS wrapped layer
        if hasattr(module, "module") and isinstance(
            module.module, (nn.Conv2d, nn.Linear)
        ):
            target = module.module
            layer_name = name + ".module"

        # CASE 2: normal layer
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            target = module
            layer_name = name

        else:
            continue

        weight = target.weight.data

        # Detect depthwise conv (MobileNet critical fix)
        if isinstance(target, nn.Conv2d) and target.groups == target.in_channels:
            bits = 8
            print(f"{layer_name} → depthwise conv → forcing 8-bit")
        else:
            bits = 4
            print(f"{layer_name} → INT4 quantized")

        target.weight.data = quantize_weight(weight, bits)

    print("\nProposed INT4 Mixed Precision completed.\n")

    return model
