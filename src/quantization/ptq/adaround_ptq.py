# src/quantization/ptq/adaround_ptq.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaRoundQuantizer:
    def __init__(self, weight, num_bits=8):
        self.weight = weight
        self.num_bits = num_bits

        qmin = -(2 ** (num_bits - 1))
        qmax = (2 ** (num_bits - 1)) - 1

        w_min = weight.min()
        w_max = weight.max()

        self.scale = (w_max - w_min) / float(qmax - qmin)
        self.zero_point = qmin - torch.round(w_min / self.scale)

        self.alpha = nn.Parameter(torch.zeros_like(weight))

    def h(self):
        return torch.clamp(torch.sigmoid(self.alpha), 0, 1)

    def quantize(self):
        w = self.weight / self.scale
        w_floor = torch.floor(w)

        w_q = w_floor + self.h()
        w_q = torch.clamp(w_q + self.zero_point, -(2**7), 2**7 - 1)

        return (w_q - self.zero_point) * self.scale


def reconstruction_loss(fp_out, quant_out):
    return F.mse_loss(quant_out, fp_out)


def apply_adaround_layer(layer, calibration_data, iters=200, lr=1e-3):

    weight = layer.weight.data.clone()
    quantizer = AdaRoundQuantizer(weight)

    optimizer = torch.optim.Adam([quantizer.alpha], lr=lr)

    layer_fp = layer

    layer_quant = type(layer)(
        layer.in_channels,
        layer.out_channels,
        layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        bias=(layer.bias is not None),
    )

    layer_quant = layer_quant.to(weight.device)

    for _ in range(iters):

        optimizer.zero_grad()

        w_q = quantizer.quantize()
        layer_quant.weight.data = w_q

        loss = 0

        for x in calibration_data:
            x = x.to(weight.device)

            with torch.no_grad():
                fp_out = layer_fp(x)

            quant_out = layer_quant(x)

            loss += reconstruction_loss(fp_out, quant_out)

        loss.backward()
        optimizer.step()

    layer.weight.data = quantizer.quantize().detach()

    return layer


def apply_adaround(model, calibration_loader, device):

    model.eval()

    calibration_data = []

    for images, _ in calibration_loader:
        calibration_data.append(images.to(device))
        if len(calibration_data) > 10:
            break

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):

            print(f"Applying AdaRound to {name}")

            apply_adaround_layer(module, calibration_data)

    return model
