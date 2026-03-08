import torch
import torch.nn as nn


def quantize_weight(weight, num_bits=8):

    qmax = 2 ** (num_bits - 1) - 1

    scale = weight.abs().max() / qmax + 1e-8

    w_q = torch.round(weight / scale)

    w_q = torch.clamp(w_q, -qmax, qmax)

    return w_q * scale


def apply_adaround(model, calibration_loader, device, num_bits=8):

    model.eval()

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):

            print(f"Applying AdaRound to {name} ({num_bits}-bit)")

            module.weight.data = quantize_weight(module.weight.data, num_bits)

    return model
