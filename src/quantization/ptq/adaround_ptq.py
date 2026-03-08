import torch
import torch.nn as nn


class AdaRoundQuantizer(nn.Module):

    def __init__(self, weight, num_bits=8):
        super().__init__()

        self.weight = weight
        self.num_bits = num_bits

        qmin = -(2 ** (num_bits - 1))
        qmax = (2 ** (num_bits - 1)) - 1

        w_min = weight.min()
        w_max = weight.max()

        self.scale = (w_max - w_min) / float(qmax - qmin + 1e-8)
        self.zero_point = qmin - torch.round(w_min / self.scale)

        self.alpha = nn.Parameter(torch.zeros_like(weight))

    def h(self):
        return torch.sigmoid(self.alpha)

    def quantize(self):

        w_scaled = self.weight / self.scale
        w_floor = torch.floor(w_scaled)

        w_q = w_floor + self.h()

        qmin = -(2 ** (self.num_bits - 1))
        qmax = (2 ** (self.num_bits - 1)) - 1

        w_q = torch.clamp(w_q + self.zero_point, qmin, qmax)

        w_dequant = (w_q - self.zero_point) * self.scale

        return w_dequant


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
