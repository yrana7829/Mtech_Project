# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class AdaRoundQuantizer(nn.Module):

#     def __init__(self, weight, num_bits=8):
#         super().__init__()

#         self.weight = weight
#         self.num_bits = num_bits

#         qmax = 2 ** (num_bits - 1) - 1
#         self.scale = weight.abs().max() / qmax + 1e-8

#         self.alpha = nn.Parameter(torch.zeros_like(weight))

#     def h(self):
#         return torch.sigmoid(self.alpha)

#     def quantize(self):

#         w_scaled = self.weight / self.scale
#         w_floor = torch.floor(w_scaled)

#         w_bar = w_floor + self.h()

#         qmax = 2 ** (self.num_bits - 1) - 1
#         qmin = -qmax - 1

#         w_bar = torch.clamp(w_bar, qmin, qmax)

#         return w_bar * self.scale


# def collect_layer_inputs(model, layer, dataloader, device, num_batches=5):

#     inputs = []

#     def hook(module, inp, out):
#         inputs.append(inp[0].detach())

#     handle = layer.register_forward_hook(hook)

#     model.eval()

#     with torch.no_grad():
#         for i, (images, _) in enumerate(dataloader):

#             if i >= num_batches:
#                 break

#             images = images.to(device)
#             model(images)

#     handle.remove()

#     return inputs


# def apply_adaround_layer(model, layer, dataloader, device, iters=200):

#     inputs = collect_layer_inputs(model, layer, dataloader, device)

#     weight = layer.weight.data.clone()

#     quantizer = AdaRoundQuantizer(weight).to(device)

#     optimizer = torch.optim.Adam([quantizer.alpha], lr=1e-3)

#     for _ in range(iters):

#         optimizer.zero_grad()

#         w_q = quantizer.quantize()

#         loss = 0

#         for x in inputs:

#             with torch.no_grad():
#                 fp_out = layer(x)

#             quant_out = F.conv2d(
#                 x,
#                 w_q,
#                 bias=layer.bias,
#                 stride=layer.stride,
#                 padding=layer.padding,
#                 dilation=layer.dilation,
#                 groups=layer.groups,
#             )

#             loss += F.mse_loss(quant_out, fp_out)

#         loss.backward()
#         optimizer.step()

#     layer.weight.data = quantizer.quantize().detach()


# def apply_adaround(model, dataloader, device):

#     model.eval()

#     for name, module in model.named_modules():

#         if isinstance(module, nn.Conv2d):

#             print(f"Applying AdaRound to {name}")

#             apply_adaround_layer(model, module, dataloader, device)

#         elif isinstance(module, nn.Linear):

#             print(f"Applying AdaRound to {name}")

#             weight = module.weight.data
#             qmax = 127
#             scale = weight.abs().max() / qmax + 1e-8

#             w_q = torch.round(weight / scale)
#             w_q = torch.clamp(w_q, -qmax, qmax)
#             module.weight.data = w_q * scale

#     return model
import torch
import torch.nn as nn


def quantize_weight(weight, num_bits=8):

    qmax = 2 ** (num_bits - 1) - 1

    scale = weight.abs().max() / qmax + 1e-8

    w_q = torch.round(weight / scale)

    w_q = torch.clamp(w_q, -qmax, qmax)

    return w_q * scale


def apply_adaround(model, calibration_loader, device):

    model.eval()

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):

            print(f"Applying AdaRound to {name}")

            module.weight.data = quantize_weight(module.weight.data)

    return model
