import torch
import torch.nn as nn


class AdaRoundQuantizer(nn.Module):

    def __init__(self, weight, num_bits=8):
        super().__init__()

        self.weight = weight
        self.num_bits = num_bits

        qmax = 2 ** (num_bits - 1) - 1

        # symmetric quantization scale
        self.scale = weight.abs().max() / qmax

        # learnable rounding parameter
        self.alpha = nn.Parameter(torch.zeros_like(weight))

    def h(self):
        return torch.sigmoid(self.alpha)

    def quantize(self):

        w_scaled = self.weight / self.scale

        w_floor = torch.floor(w_scaled)

        # adaptive rounding
        w_bar = w_floor + self.h()

        qmax = 2 ** (self.num_bits - 1) - 1
        qmin = -qmax - 1

        w_bar = torch.clamp(w_bar, qmin, qmax)

        # dequantize
        w_q = w_bar * self.scale

        return w_q


def apply_adaround_layer(layer, iters=200, lr=1e-3):

    weight = layer.weight.data.clone()

    quantizer = AdaRoundQuantizer(weight)

    optimizer = torch.optim.Adam([quantizer.alpha], lr=lr)

    for _ in range(iters):

        optimizer.zero_grad()

        w_q = quantizer.quantize()

        loss = torch.mean((w_q - weight) ** 2)

        loss.backward()
        optimizer.step()

    layer.weight.data = quantizer.quantize().detach()

    return layer


def apply_adaround(model, calibration_loader, device):

    model.eval()

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):

            print(f"Applying AdaRound to {name}")

            apply_adaround_layer(module)

    return model
