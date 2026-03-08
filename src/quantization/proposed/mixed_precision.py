import torch
import torch.nn as nn


def quantize_weight(weight, num_bits):

    qmax = 2 ** (num_bits - 1) - 1

    scale = weight.abs().max() / qmax + 1e-8

    q = torch.round(weight / scale)

    q = torch.clamp(q, -qmax, qmax)

    return q * scale


def apply_mixed_precision(model):

    print("\nApplying Mixed Precision Quantization...\n")

    first_conv = True

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            weight = module.weight.data

            # First convolution layer
            if first_conv:
                bits = 8
                first_conv = False

            # Depthwise convolution
            elif module.groups == module.in_channels:
                bits = 6

            else:
                bits = 8

            module.weight.data = quantize_weight(weight, bits)

            print(f"{name}  → {bits}-bit")

        elif isinstance(module, nn.Linear):

            weight = module.weight.data

            bits = 8

            module.weight.data = quantize_weight(weight, bits)

            print(f"{name}  → {bits}-bit")

    return model
