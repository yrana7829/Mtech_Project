import torch
import torch.nn as nn


def quantize_weight(weight, num_bits):

    qmax = 2 ** (num_bits - 1) - 1

    scale = weight.abs().max() / qmax + 1e-8

    q = torch.round(weight / scale)
    q = torch.clamp(q, -qmax, qmax)

    return q * scale


def apply_proposed_mixed_precision(model):

    print("\nApplying Proposed Mixed Precision...\n")

    for name, module in model.named_modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            weight = module.weight.data

            variance = weight.var()

            # sensitivity rule
            if variance > 0.02:
                bits = 8
            else:
                bits = 6

            module.weight.data = quantize_weight(weight, bits)

            print(f"{name} → variance={variance.item():.6f} → {bits}-bit")

    print("\nProposed Mixed Precision completed.\n")

    return model
