import torch
import torch.nn as nn


def quantize_weight(weight, num_bits):

    qmax = 2 ** (num_bits - 1) - 1

    scale = weight.abs().max() / qmax + 1e-8

    q = torch.round(weight / scale)
    q = torch.clamp(q, -qmax, qmax)

    return q * scale


def apply_proposed_mixed_precision_int4(model):

    print("\nApplying Proposed Mixed Precision INT4...\n")

    for name, module in model.named_modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            weight = module.weight.data

        elif hasattr(module, "module") and isinstance(
            module.module, (nn.Conv2d, nn.Linear)
        ):

            weight = module.module.weight.data

        else:
            continue

        bits = 4

        quant_weight = quantize_weight(weight, bits)

        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.weight.data = quant_weight
        else:
            module.module.weight.data = quant_weight

        print(f"{name} → INT4 quantized")

    print("\nProposed INT4 Mixed Precision completed.\n")

    return model
