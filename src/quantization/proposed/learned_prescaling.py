import torch
import torch.nn as nn


def apply_learned_prescaling(model, device):

    model.eval()
    model.to(device)

    print("\nApplying Learned Pre-Scaling...\n")

    for name, module in model.named_modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            weight = module.weight.data

            # compute normalization scale
            scale = weight.abs().mean() + 1e-8

            # normalize weights
            module.weight.data = weight / scale

            # store scale for later use
            module.register_buffer("lps_scale", scale)

            print(f"{name} → normalize_scale={scale.item():.6f}")

    print("\nLearned Pre-Scaling completed.\n")

    return model


class LPSWrapper(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.scale = module.lps_scale

    def forward(self, x):
        return self.module(x) * self.scale


def wrap_lps_layers(model):

    for name, module in model.named_children():

        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "lps_scale"):

            setattr(model, name, LPSWrapper(module))

        else:
            wrap_lps_layers(module)

    return model
