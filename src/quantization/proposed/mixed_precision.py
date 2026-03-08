import torch
import torch.nn as nn


def quantize_weight(weight, bits):

    qmax = 2 ** (bits - 1) - 1

    scale = weight.abs().max() / qmax + 1e-8

    q = torch.round(weight / scale)

    q = torch.clamp(q, -qmax, qmax)

    return q * scale


def apply_mixed_precision(model):

    print("\nApplying Mixed Precision Allocation...\n")

    first_conv = True

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            weight = module.weight.data

            if first_conv:
                bits = 8
                first_conv = False

            elif module.groups == module.in_channels:
                bits = 6
            else:
                bits = 8

            module.weight.data = quantize_weight(weight, bits)

            print(f"{name} → {bits}-bit")

        elif isinstance(module, nn.Linear):

            module.weight.data = quantize_weight(module.weight.data, 8)

            print(f"{name} → 8-bit")

    print("\nMixed Precision Allocation completed.\n")

    return model
