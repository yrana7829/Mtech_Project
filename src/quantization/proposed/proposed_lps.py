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
            scale = torch.sqrt(torch.mean(weight**2)) + 1e-8

            module.weight.data = weight / scale

            module.register_buffer("proposed_lps_scale", scale)

            print(f"{name} → rms_scale={scale.item():.6f}")

    print("\nProposed LPS completed.\n")

    return model
