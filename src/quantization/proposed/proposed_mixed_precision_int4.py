import torch
import torch.nn as nn


def quantize_weight(weight, num_bits):

    qmax = 2 ** (num_bits - 1) - 1

    # Robust scale (percentile instead of max)
    scale = torch.quantile(weight.abs(), 0.999) / qmax + 1e-8

    q = torch.round(weight / scale)
    q = torch.clamp(q, -qmax, qmax)

    return q * scale


def apply_proposed_mixed_precision_int4(model):

    print("\nApplying Proposed Mixed Precision INT4...\n")

    for name, module in model.named_modules():

        # Case 1: normal Conv/Linear layer
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data

        # Case 2: LPS wrapped layer
        elif hasattr(module, "module") and isinstance(
            module.module, (nn.Conv2d, nn.Linear)
        ):
            weight = module.module.weight.data

        else:
            continue

        # Detect depthwise convolution
        if isinstance(module, nn.Conv2d) and module.groups == module.in_channels:
            bits = 8
            print(f"{name} → depthwise conv → forcing 8-bit")

        else:
            bits = 4
            print(f"{name} → INT4 quantized")

        quant_weight = quantize_weight(weight, bits)

        # assign weight back correctly
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.weight.data = quant_weight
        else:
            module.module.weight.data = quant_weight

    print("\nProposed INT4 Mixed Precision completed.\n")

    return model
