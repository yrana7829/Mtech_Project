import torch
import torch.nn as nn


def quantize_weight(weight, num_bits=8):

    qmax = 2 ** (num_bits - 1) - 1

    scale = weight.abs().max() / qmax + 1e-8

    q = torch.round(weight / scale)
    q = torch.clamp(q, -qmax, qmax)

    return q * scale


def apply_naive_ptq(model):

    print("\nApplying Naive PTQ...\n")

    for name, module in model.named_modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            weight = module.weight.data
            quant_weight = quantize_weight(weight)

            # restore original scale if LPS was applied
            if hasattr(module, "lps_scale"):
                quant_weight = quant_weight * module.lps_scale
            module.weight.data = quant_weight

            print(f"Quantized {name}")

    return model
