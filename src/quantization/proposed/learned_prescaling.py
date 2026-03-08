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

            # store scale for restoration after quantization
            module.register_buffer("lps_scale", scale)

            print(f"{name}  → normalize_scale={scale.item():.4f}")

    print("\nLearned Pre-Scaling completed.\n")

    return model
