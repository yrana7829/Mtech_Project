import torch
import torch.nn as nn


def quantize_weight(weight, num_bits=8):

    qmax = 2 ** (num_bits - 1) - 1

    scale = weight.abs().max() / qmax + 1e-8

    w_q = torch.round(weight / scale)

    w_q = torch.clamp(w_q, -qmax, qmax)

    w_dequant = w_q * scale

    return w_dequant


def apply_adaround(model, calibration_loader, device):

    model.eval()

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):

            print(f"Applying AdaRound to {name}")

            weight = module.weight.data

            module.weight.data = quantize_weight(weight)

    return model
