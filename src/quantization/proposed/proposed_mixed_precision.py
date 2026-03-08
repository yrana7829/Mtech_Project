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

        variance = weight.var()

        # sensitivity rule
        if variance > 0.02:
            bits = 8
        else:
            bits = 6

        quant_weight = quantize_weight(weight, bits)

        # assign weight back correctly
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.weight.data = quant_weight
        else:
            module.module.weight.data = quant_weight

        print(f"{name} → variance={variance.item():.6f} → {bits}-bit")

    print("\nProposed Mixed Precision completed.\n")

    return model
