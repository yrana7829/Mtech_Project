import torch
import torch.nn as nn


def apply_proposed_lps(model, device):

    model.eval()
    model.to(device)

    print("\nApplying Proposed Learned Pre-Scaling (RMS)...\n")

    for name, module in model.named_modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            weight = module.weight.data

            # RMS scaling
            scale = torch.sqrt(torch.mean(weight**2) + 1e-8)

            # normalize weights
            module.weight.data = weight / scale

            # store scale
            module.register_buffer("proposed_lps_scale", scale)

            print(f"{name} → rms_scale={scale.item():.6f}")

    print("\nProposed LPS completed.\n")

    return model


class ProposedLPSWrapper(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.scale = module.proposed_lps_scale

    def forward(self, x):
        return self.module(x) * self.scale


def wrap_proposed_lps_layers(model):

    for name, module in model.named_children():

        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(
            module, "proposed_lps_scale"
        ):

            setattr(model, name, ProposedLPSWrapper(module))

        else:
            wrap_proposed_lps_layers(module)

    return model
